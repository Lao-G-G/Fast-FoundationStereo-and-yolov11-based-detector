"""
Stereo depth estimator using Fast-FoundationStereo.
Replaces FoundationStereo with the faster real-time variant for higher throughput.
"""
import os
import sys

# Must disable torch.compile BEFORE importing torch or loading model,
# because the serialized model contains @torch.compile-decorated functions
# that fail when Triton is not installed.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import cv2
import numpy as np
import torch
import logging

# Add Fast-FoundationStereo paths (configurable via FFS_ROOT env var)
code_dir = os.path.dirname(os.path.realpath(__file__))
ffs_root = os.environ.get('FFS_ROOT', os.path.join(code_dir, '..'))
ffs_root = os.path.abspath(ffs_root)
sys.path.insert(0, ffs_root)
sys.path.insert(0, os.path.join(ffs_root, 'core'))

from core.utils.utils import InputPadder


class StereoDepthEstimator:
    """
    Stereo depth estimation using Fast-FoundationStereo.
    Takes left+right image pair and produces metric depth map.
    Runs 10x+ faster than original FoundationStereo with comparable zero-shot accuracy.
    """

    def __init__(self, ckpt_dir, camera_K=None, baseline=0.54,
                 device=None, scale=1.0, valid_iters=8):
        """
        Initialize the stereo depth estimator.

        Args:
            ckpt_dir (str): Path to Fast-FoundationStereo pretrained model (.pth file).
                            Fast-FoundationStereo uses serialized models (e.g. model_best_bp2_serialize.pth).
            camera_K (numpy.ndarray): Camera intrinsic matrix (3x3). If None, uses KITTI default.
            baseline (float): Stereo baseline in meters (used to convert disparity to depth).
            device (str): Device to run inference on ('cuda', 'cpu').
            scale (float): Image downsample scale (<=1.0).
            valid_iters (int): Number of GRU update iterations (default 8 for real-time).
        """
        self._init_device(device)
        self.baseline = baseline
        self.scale = scale
        self.valid_iters = valid_iters

        # Set default KITTI camera intrinsics if not provided
        if camera_K is None:
            # KITTI default intrinsics (after cropping to ~1242x375)
            self.K = np.array([
                [718.856, 0.0, 607.1928],
                [0.0, 718.856, 185.2157],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        else:
            self.K = camera_K.astype(np.float32)

        self.focal_length = self.K[0, 0]

        # Load Fast-FoundationStereo model (serialized, not state dict)
        self._load_model(ckpt_dir)

    def _init_device(self, device):
        """Set up compute device."""
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        print(f"[FastStereoDepth] Using device: {self.device}")

    def _load_model(self, ckpt_dir):
        """Load Fast-FoundationStereo serialized model."""
        ckpt_dir = os.path.abspath(ckpt_dir)
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

        print(f"[FastStereoDepth] Loading Fast-FoundationStereo model from {ckpt_dir}...")
        print(f"[FastStereoDepth] Note: 1st inference will be slow due to CUDA JIT compilation.")

        # Fast-FoundationStereo uses torch.load() directly (serialized entire model)
        self.model = torch.load(ckpt_dir, map_location='cpu', weights_only=False)

        # Override with our settings
        self.model.args.valid_iters = self.valid_iters

        if self.device.startswith('cuda'):
            self.model.cuda()
        else:
            self.model.cpu()
        self.model.eval()

        print("[FastStereoDepth] Model loaded successfully.")

    def estimate_depth(self, left_image, right_image):
        """
        Estimate metric depth from a stereo pair.

        Args:
            left_image (numpy.ndarray): Left image (BGR, HxWx3, uint8)
            right_image (numpy.ndarray): Right image (BGR, HxWx3, uint8)

        Returns:
            numpy.ndarray: Metric depth map in meters (H x W, float32)
        """
        H, W = left_image.shape[:2]

        # Convert BGR to RGB and resize
        if self.scale < 1.0:
            left_rgb = cv2.resize(left_image, None, fx=self.scale, fy=self.scale)
            right_rgb = cv2.resize(right_image, None, fx=self.scale, fy=self.scale)
        else:
            left_rgb = left_image.copy()
            right_rgb = right_image.copy()

        left_rgb = cv2.cvtColor(left_rgb, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_rgb, cv2.COLOR_BGR2RGB)

        # Convert to tensors (B, C, H, W), range 0-255
        img0 = torch.as_tensor(left_rgb).float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb).float()[None].permute(0, 3, 1, 2)

        if self.device.startswith('cuda'):
            img0 = img0.cuda()
            img1 = img1.cuda()

        # Pad for network
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        # Run inference
        with torch.no_grad():
            if self.device.startswith('cuda'):
                with torch.amp.autocast('cuda', enabled=True):
                    disp = self.model.forward(
                        img0, img1,
                        iters=self.valid_iters,
                        test_mode=True,
                        optimize_build_volume='pytorch1'
                    )
            else:
                disp = self.model.forward(
                    img0, img1,
                    iters=self.valid_iters,
                    test_mode=True,
                    optimize_build_volume='pytorch1'
                )

        # Unpad and convert to numpy
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)

        # Convert disparity to metric depth: depth = focal_length * baseline / disparity
        # Clamp disparity to avoid division by zero
        disp_safe = np.clip(disp, 0.001, None)
        depth_meters = self.focal_length * self.baseline / disp_safe

        # Handle invalid disparities
        depth_meters[disp <= 0] = 0.0

        return depth_meters.astype(np.float32)

    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """
        Colorize depth map for visualization.

        Args:
            depth_map (numpy.ndarray): Depth map (metric or normalized)
            cmap (int): OpenCV colormap

        Returns:
            numpy.ndarray: Colorized depth map (BGR)
        """
        # If metric depth, normalize first
        valid = depth_map > 0
        if np.any(valid):
            d_min = depth_map[valid].min()
            d_max = depth_map[valid].max()
            if d_max > d_min:
                depth_norm = np.zeros_like(depth_map)
                depth_norm[valid] = (depth_map[valid] - d_min) / (d_max - d_min)
                depth_uint8 = (depth_norm * 255).astype(np.uint8)
            else:
                depth_uint8 = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            depth_uint8 = np.zeros_like(depth_map, dtype=np.uint8)

        colored = cv2.applyColorMap(depth_uint8, cmap)
        return colored

    def get_depth_at_point(self, depth_map, x, y):
        """
        Get depth value at a specific point.

        Args:
            depth_map (numpy.ndarray): Depth map (metric or normalized)
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            float: Depth value at (x, y)
        """
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return float(depth_map[y, x])
        return 0.0

    def get_depth_in_region(self, depth_map, bbox, method='median'):
        """
        Get depth value in a region defined by a bounding box.

        Args:
            depth_map (numpy.ndarray): Depth map
            bbox (list): Bounding box [x1, y1, x2, y2]
            method (str): 'median', 'mean', 'min'

        Returns:
            float: Depth value in the region
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)

        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0

        # Only consider valid (non-zero) depth values
        valid_region = region[region > 0]
        if valid_region.size == 0:
            return 0.0

        if method == 'median':
            return float(np.median(valid_region))
        elif method == 'mean':
            return float(np.mean(valid_region))
        elif method == 'min':
            return float(np.min(valid_region))
        else:
            return float(np.median(valid_region))