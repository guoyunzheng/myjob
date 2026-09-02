from kornia import augmentation as K
import torch

from .base import DataPreprocessor


class PeractDataPreprocessor(DataPreprocessor):

    def __init__(self, keypose_only=False, num_history=1,
                 orig_imsize=256, custom_imsize=None, depth2cloud=None):
        super().__init__(
            keypose_only=keypose_only,
            num_history=num_history,
            custom_imsize=custom_imsize,
            depth2cloud=depth2cloud
        )
        # RGB and XYZ must share exactly the same sampled image transform, but
        # they must not share an interpolation rule. Bilinear interpolation is
        # appropriate for color; on a point cloud it invents 3D points between
        # foreground and background surfaces.
        self.rgb_aug = K.AugmentationSequential(
            K.RandomAffine(
                degrees=0,
                translate=0.0,
                scale=(0.90, 1.10),
                resample="bilinear",
                padding_mode="reflection",
                p=0.5
            ),
            K.RandomResizedCrop(
                size=(orig_imsize, orig_imsize),
                scale=(0.95, 1.0),
                resample="bilinear",
                p=0.1
            )
        ).cuda()
        self.pcd_aug = K.AugmentationSequential(
            K.RandomAffine(
                degrees=0,
                translate=0.0,
                scale=(0.90, 1.10),
                resample="nearest",
                padding_mode="reflection",
                p=0.5
            ),
            K.RandomResizedCrop(
                size=(orig_imsize, orig_imsize),
                scale=(0.95, 1.0),
                resample="nearest",
                # The slice crop ends in F.interpolate, where nearest mode
                # requires align_corners=None (Kornia defaults to True).
                align_corners=None,
                p=0.1
            )
        ).cuda()

    def process_obs(self, rgbs, pcds, augment=False):
        """
        RGBs of shape (B, ncam, 3, h_i, w_i),
        depths of shape (B, ncam, h_i, w_i).
        Assume the 3d cameras go before 2d cameras.
        """
        # Handle non-wrist cameras, which may require augmentations
        if augment:
            b, nc, _, h, w = rgbs.shape
            rgb_flat = (
                rgbs.cuda(non_blocking=True).float() / 255
            ).reshape(-1, 3, h, w)
            pcd_flat = pcds.cuda(non_blocking=True).float().reshape(
                -1, 3, h, w
            )

            rgb_3d = self.rgb_aug(rgb_flat)
            # AugmentationSequential exposes the sampled parameter list so the
            # geometrically identical transform can be replayed with nearest
            # interpolation on XYZ.
            pcd_3d = self.pcd_aug(
                pcd_flat,
                params=self.rgb_aug._params,
            )
            rgb_3d = rgb_3d.reshape(b, nc, 3, h, w)
            pcd_3d = pcd_3d.reshape(b, nc, 3, h, w)
        else:
            # Simply convert to full precision
            rgb_3d = rgbs.cuda(non_blocking=True).float() / 255
            pcd_3d = pcds.cuda(non_blocking=True).float()

        return rgb_3d, pcd_3d
