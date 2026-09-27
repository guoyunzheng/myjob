from utils.action_contract import select_proprio_history


class DataPreprocessor:

    def __init__(self, keypose_only=False, num_history=1,
                 custom_imsize=None, depth2cloud=None):
        self.keypose_only = keypose_only
        self.num_history = num_history
        self.custom_imsize = custom_imsize
        self.depth2cloud = depth2cloud

    def process_actions(self, actions):
        """Action shape: (B, T, nhand, 3+rot+1)."""
        actions = actions.cuda(non_blocking=True)
        if self.keypose_only:
            actions = actions[:, [-1]]
        return actions

    def process_proprio(self, proprio):
        """Proprio shape: (B, nhist, nhand, 3+rot+1)."""
        proprio = proprio.cuda(non_blocking=True)
        # Zarr stores [oldest, ..., current], as does online observation history.
        # Taking the prefix would drop the current pose for num_history=1/2.
        return select_proprio_history(proprio, self.num_history)

    def process_obs(self, rgbs, pcds):
        pass
