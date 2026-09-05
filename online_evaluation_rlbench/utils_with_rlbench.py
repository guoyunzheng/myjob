import os
import glob
import random

import open3d  # DON'T DELETE THIS!
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from modeling.encoder.text import fetch_tokenizers
from online_evaluation_rlbench.get_stored_demos import get_stored_demos
from online_evaluation_rlbench.gripper_control import (
    hysteresis_gripper_command,
    should_execute_gripper_change,
)


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


class Mover:

    def __init__(
        self,
        task,
        max_tries=1,
        initial_action=None,
        gripper_close_threshold=0.25,
        gripper_open_threshold=0.75,
        require_pose_for_gripper_open=True,
    ):
        self._task = task
        self._last_action = (
            None if initial_action is None else initial_action.copy()
        )
        self._max_tries = max_tries
        self._gripper_close_threshold = gripper_close_threshold
        self._gripper_open_threshold = gripper_open_threshold
        self._require_pose_for_gripper_open = (
            require_pose_for_gripper_open
        )
        self.last_position_error = float("nan")
        self.last_rotation_error_deg = float("nan")
        self.last_attempts = 0
        self.last_pose_reached = False
        self.last_gripper_probability = float("nan")
        self.last_gripper_command = float("nan")
        self.last_gripper_change_requested = False
        self.last_gripper_change_executed = False
        self.last_gripper_change_suppressed = False

    @staticmethod
    def _rotation_error_deg(target_quat, actual_quat):
        target_quat = target_quat / np.clip(
            np.linalg.norm(target_quat), 1e-8, None
        )
        actual_quat = actual_quat / np.clip(
            np.linalg.norm(actual_quat), 1e-8, None
        )
        cosine = np.clip(np.abs(np.dot(target_quat, actual_quat)), 0.0, 1.0)
        return float(np.degrees(2.0 * np.arccos(cosine)))

    def __call__(self, action, collision_checking=False):
        # action is an array (8,)
        obs = None
        terminate = None
        reward = 0

        # Resolve the probabilistic output while preserving the measured state
        # inside an uncertainty band. Evaluation normally supplies an initial
        # action from the reset observation; retain a safe fallback for callers
        # that do not.
        target = action.copy()
        self.last_gripper_probability = float(np.clip(target[7], 0.0, 1.0))
        current_gripper = (
            float(self._last_action[7])
            if self._last_action is not None
            else (1.0 if self.last_gripper_probability >= 0.5 else 0.0)
        )
        target_gripper = hysteresis_gripper_command(
            self.last_gripper_probability,
            current_gripper,
            close_threshold=self._gripper_close_threshold,
            open_threshold=self._gripper_open_threshold,
        )
        target[7] = target_gripper
        self.last_gripper_command = target_gripper
        self.last_gripper_change_requested = target_gripper != current_gripper
        self.last_gripper_change_executed = False
        self.last_gripper_change_suppressed = False

        # Reach the desired arm pose while holding the current gripper state.
        action = target.copy()
        action[7] = current_gripper
        self.last_pose_reached = False
        for attempt in range(1, self._max_tries + 1):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

            # Check if we reached the desired pose (planner may be inaccurate)
            pos = obs.gripper_pose[:3]
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())
            dist_rot = self._rotation_error_deg(
                target[3:7], obs.gripper_pose[3:7]
            )
            self.last_position_error = float(dist_pos)
            self.last_rotation_error_deg = dist_rot
            self.last_attempts = attempt
            criteria = (dist_pos < 5e-3, dist_rot < 5.0)

            if all(criteria) or reward == 1:
                self.last_pose_reached = all(criteria)
                break

        # Delay only closed->open until the requested pose is reached. Closing
        # must remain available for grasping even if the pose tolerance misses.
        allow_gripper_change = should_execute_gripper_change(
            current_gripper,
            target_gripper,
            self.last_pose_reached,
            require_pose_for_opening=self._require_pose_for_gripper_open,
        )
        if (
            reward != 1.0
            and self.last_gripper_change_requested
            and allow_gripper_change
        ):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = target
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)
            self.last_gripper_change_executed = True
        elif self.last_gripper_change_requested and not allow_gripper_change:
            self.last_gripper_change_suppressed = True

        # Track the actually observed state, not an unexecuted target command.
        stored_action = target.copy()
        if obs is not None:
            stored_action[7] = float(obs.gripper_open)
        else:
            stored_action[7] = current_gripper
        self._last_action = stored_action

        return obs, reward, terminate


class Actioner:

    def __init__(self, policy=None, backbone='clip'):
        self._policy = policy.cuda()
        self._policy.eval()
        self._instr = None
        self.tokenizer = fetch_tokenizers(backbone)

    def load_episode(self, descriptions):
        instr = [random.choice(descriptions)]
        self._instr = self.tokenizer(instr).cuda(non_blocking=True)

    def predict(self, rgbs, pcds, gripper, prediction_len=1):
        """
        Args:
            rgbs: (1, ncam, 3, H, W)
            pcds: (1, ncam, 3, H, W)
            gripper: (1, nhist, 8)
            prediction_len: int

        Returns:
            (1, prediction_len, 8)
        """
        return self._policy(
            None,
            torch.full([1, prediction_len, 1], False).cuda(non_blocking=True),
            rgbs,
            None,
            pcds,
            self._instr,
            gripper[:, :, None, :],
            run_inference=True
        ).view(1, prediction_len, 8)


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        task_str=None,
        image_size=(256, 256),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        collision_checking=False,
        gripper_close_threshold=0.25,
        gripper_open_threshold=0.75,
        require_pose_for_gripper_open=True,
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras
        self.collision_checking = collision_checking
        if not 0.0 <= gripper_close_threshold < gripper_open_threshold <= 1.0:
            raise ValueError(
                "Expected 0 <= gripper_close_threshold < "
                "gripper_open_threshold <= 1."
            )
        self.gripper_close_threshold = gripper_close_threshold
        self.gripper_open_threshold = gripper_open_threshold
        self.require_pose_for_gripper_open = require_pose_for_gripper_open

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=Discrete()
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless
        )
        self.image_size = image_size

    def get_obs_action(self, obs):
        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        obs_rgb = [
            torch.tensor(state_dict["rgb"][i]).float().permute(2, 0, 1) / 255.0
            for i in range(len(state_dict["rgb"]))
        ]
        obs_pc = [
            torch.tensor(state_dict["pc"][i]).float().permute(2, 0, 1)
            if len(state_dict["pc"]) > 0 else None
            for i in range(len(state_dict["rgb"]))
        ]
        state = torch.cat(obs_rgb + obs_pc, dim=0)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        return rgb, pcd, gripper

    def evaluate_task_on_multiple_variations(
        self,
        task_str,
        max_steps,
        actioner,
        max_tries=1,
        prediction_len=1,
        num_history=1
    ):
        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = glob.glob(
            os.path.join(self.data_path, task_str, "variation*")
        )
        task_variations = [
            int(n.split('/')[-1].replace('variation', ''))
            for n in task_variations
        ]

        var_success_rates = {}
        var_success_counts = {}
        var_num_valid_demos = {}

        for variation in tqdm(task_variations):
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    actioner=actioner,
                    max_tries=max_tries,
                    prediction_len=prediction_len,
                    num_history=num_history
                )
            )
            if valid:
                var_success_counts[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos
                var_success_rates[variation] = success_rate / num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_counts.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str,  # this is str
        task,  # instance of TaskEnvironment
        max_steps,
        variation,
        actioner,
        max_tries=1,
        prediction_len=1,
        num_history=1
    ):
        success_rate = 0
        total_reward = 0
        var_demos = get_stored_demos(
            amount=-1,
            dataset_root=self.data_path,
            variation_number=variation,
            task_name=task_str,
            random_selection=False,
            from_episode_number=0
        )

        for demo_id, demo in enumerate(var_demos):

            grippers = torch.Tensor([]).cuda(non_blocking=True)
            descriptions, obs = task.reset_to_demo(demo)
            actioner.load_episode(descriptions)

            initial_action = np.concatenate(
                (obs.gripper_pose, [obs.gripper_open])
            )
            move = Mover(
                task,
                max_tries=max_tries,
                initial_action=initial_action,
                gripper_close_threshold=self.gripper_close_threshold,
                gripper_open_threshold=self.gripper_open_threshold,
                require_pose_for_gripper_open=(
                    self.require_pose_for_gripper_open
                ),
            )
            max_reward = 0.0
            execution_position_errors = []
            execution_rotation_errors = []
            execution_attempts = []
            num_action_errors = 0
            gripper_probabilities = []
            gripper_changes = 0
            suppressed_gripper_changes = 0

            for step_id in range(max_steps):

                # Fetch the current observation, and predict one action
                rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                rgbs_input = rgb.cuda(non_blocking=True)
                pcds_input = pcd.cuda(non_blocking=True)
                gripper = gripper.cuda(non_blocking=True)
                grippers = torch.cat([grippers, gripper.unsqueeze(1)], 1)

                # Prepare proprioception history
                gripper_input = grippers[:, -num_history:]
                npad = num_history - gripper_input.shape[1]
                gripper_input = F.pad(
                    gripper_input, (0, 0, npad, 0), mode='replicate'
                )

                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    prediction_len=prediction_len
                )

                # Update the observation based on the predicted action
                try:
                    # Execute entire predicted trajectory step by step
                    actions = output[-1].cpu().numpy()

                    # execute
                    for action in actions:
                        obs, reward, _ = move(
                            action,
                            collision_checking=self.collision_checking,
                        )
                        execution_position_errors.append(move.last_position_error)
                        execution_rotation_errors.append(move.last_rotation_error_deg)
                        execution_attempts.append(move.last_attempts)
                        gripper_probabilities.append(
                            move.last_gripper_probability
                        )
                        gripper_changes += int(
                            move.last_gripper_change_executed
                        )
                        suppressed_gripper_changes += int(
                            move.last_gripper_change_suppressed
                        )
                        if move.last_gripper_change_suppressed:
                            print(
                                task_str,
                                "gripper change suppressed",
                                f"step={step_id}",
                                f"p_open={move.last_gripper_probability:.3f}",
                                "pose_error="
                                f"{1e3 * move.last_position_error:.1f}mm/"
                                f"{move.last_rotation_error_deg:.1f}deg",
                            )

                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        success_rate += 1
                        break

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    num_action_errors += 1
                    reward = 0

            total_reward += max_reward

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id + 1}",
                f"SR: {total_reward:.2f}/{demo_id + 1}",
                "# valid demos", demo_id + 1,
                "PolicySteps", step_id + 1,
                "ActionErrors", num_action_errors,
                "PlanErr(mm/deg)",
                (
                    f"{1e3 * np.mean(execution_position_errors):.1f}/"
                    f"{np.mean(execution_rotation_errors):.1f}"
                    if execution_position_errors else "n/a"
                ),
                "PlanAttempts",
                (
                    f"{np.mean(execution_attempts):.2f}"
                    if execution_attempts else "n/a"
                ),
                "GripperP(min/mean/max)",
                (
                    f"{np.min(gripper_probabilities):.3f}/"
                    f"{np.mean(gripper_probabilities):.3f}/"
                    f"{np.max(gripper_probabilities):.3f}"
                    if gripper_probabilities else "n/a"
                ),
                "GripperChanges/Suppressed",
                f"{gripper_changes}/{suppressed_gripper_changes}",
            )

        # Compensate for failed demos
        valid = len(var_demos) > 0

        return success_rate, valid, len(var_demos)

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras,
        **kwargs
    ):
        # Define a config for an unused camera with all applications as False.
        unused_cams = CameraConfig()
        unused_cams.set_all(False)

        # Define a config for a used camera with the given image size and flags
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL,
            **kwargs
        )

        # apply_cameras is a tuple with the names(str) of all the cameras
        camera_names = apply_cameras
        kwargs = {}
        for n in camera_names:
            kwargs[n] = used_cams

        obs_config = ObservationConfig(
            front_camera=kwargs.get("front", unused_cams),
            left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
            right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
            wrist_camera=kwargs.get("wrist", unused_cams),
            overhead_camera=kwargs.get("overhead", unused_cams),
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True
        )

        return obs_config
