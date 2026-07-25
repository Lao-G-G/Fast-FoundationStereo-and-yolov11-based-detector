#!/usr/bin/env python3
"""
LiDAR-Stereo depth fusion for 3D object detection.
Projects LiDAR point cloud onto the left camera image and fuses with stereo depth
inside each 2D bbox to improve depth accuracy.
"""
import os
import sys
import numpy as np
import cv2

# KITTI default calibration (camera 2 = left color camera)
DEFAULT_R0_RECT = np.eye(3, dtype=np.float32)
DEFAULT_TR_VELO_TO_CAM = np.array([
    [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
    [1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
    [9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01]
], dtype=np.float32)


def load_kitti_calib(calib_path):
    """Load KITTI calib file. Returns P2 (3x4), R0_rect (3x3), Tr_velo_to_cam (3x4)."""
    P2 = None
    R0_rect = DEFAULT_R0_RECT.copy()
    Tr_velo_cam = DEFAULT_TR_VELO_TO_CAM.copy()

    if not os.path.exists(calib_path):
        return P2, R0_rect, Tr_velo_cam

    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('P2:'):
                P2 = np.array([float(x) for x in line.split(':')[1].split()]).reshape(3, 4)
            elif line.startswith('R0_rect:'):
                R0_rect = np.array([float(x) for x in line.split(':')[1].split()]).reshape(3, 3)
            elif line.startswith('Tr_velo_to_cam:'):
                Tr_velo_cam = np.array([float(x) for x in line.split(':')[1].split()]).reshape(3, 4)

    if P2 is None:
        P2 = np.array([
            [718.856, 0.0, 607.1928, 0.0],
            [0.0, 718.856, 185.2157, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float32)

    return P2.astype(np.float32), R0_rect.astype(np.float32), Tr_velo_cam.astype(np.float32)


def load_lidar_bin(bin_path):
    """Load KITTI .bin point cloud. Returns Nx3 (x,y,z) array."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def project_lidar_to_camera(points_velo, R0_rect, Tr_velo_to_cam, K):
    """Project LiDAR points to camera image plane. Returns uv (N,2), z_cam (N,)."""
    N = points_velo.shape[0]
    pts_velo_h = np.hstack([points_velo, np.ones((N, 1))])[:, :, None]
    pts_cam = (R0_rect @ Tr_velo_to_cam) @ pts_velo_h
    pts_cam = pts_cam[:, :, 0]
    pts_img = K @ pts_cam.T
    z_cam = pts_img[2, :]
    u = pts_img[0, :] / (z_cam + 1e-8)
    v = pts_img[1, :] / (z_cam + 1e-8)
    return np.stack([u, v], axis=1), z_cam


def fuse_depth_in_bbox(stereo_depth, lidar_depth, lidar_n,
                       stereo_weight=0.3, min_lidar_points=5, max_fuse_dist=0.5):
    """
    Fuse stereo with LiDAR depth for one bbox.
    - lidar_n < min_lidar_points → keep stereo
    - |stereo - lidar| < max_fuse_dist → weighted average
    - otherwise → trust LiDAR
    """
    if lidar_n < min_lidar_points or lidar_depth <= 0:
        return stereo_depth, 'stereo'
    if abs(stereo_depth - lidar_depth) > max_fuse_dist:
        # Large discrepancy → LiDAR projection likely wrong, keep stereo
        return stereo_depth, 'stereo'
    # Small discrepancy → weighted fusion, binocular is dominant
    fused = stereo_weight * stereo_depth + (1 - stereo_weight) * lidar_depth
    return fused, 'fused'


def get_lidar_depth_in_bbox(uv_points, z_points, bbox, H, W, min_depth=0.5, max_depth=100.0):
    """Median LiDAR depth inside a 2D bbox."""
    x1, y1, x2, y2 = [int(c) for c in bbox]
    x1, x2 = max(0, min(x1, W - 1)), max(0, min(x2, W - 1))
    y1, y2 = max(0, min(y1, H - 1)), max(0, min(y2, H - 1))

    u, v = uv_points[:, 0], uv_points[:, 1]
    in_bbox = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    in_range = (z_points > min_depth) & (z_points < max_depth)
    valid = in_bbox & in_range
    if valid.sum() == 0:
        return 0.0, 0
    return float(np.median(z_points[valid])), int(valid.sum())


class LidarDepthFusion:
    """LiDAR-Stereo depth fusion module."""

    def __init__(self, camera_K, baseline=0.54,
                 calib_path=None, calib_velo_path=None,
                 stereo_weight=0.3, min_lidar_points=5, max_fuse_dist=0.5):
        self.K = camera_K.astype(np.float32)
        self.baseline = baseline
        self.stereo_weight = stereo_weight
        self.min_lidar_points = min_lidar_points
        self.max_fuse_dist = max_fuse_dist

        # P2 and R0_rect
        if calib_path and os.path.exists(calib_path):
            self.P2, self.R0_rect, _ = load_kitti_calib(calib_path)
        else:
            self.P2 = np.hstack([self.K, np.zeros((3, 1))])
            self.R0_rect = DEFAULT_R0_RECT.copy()

        # Tr_velo_to_cam
        if calib_velo_path and os.path.exists(calib_velo_path):
            _, _, self.Tr_velo_to_cam = load_kitti_calib(calib_velo_path)
        elif calib_path and os.path.exists(calib_path):
            _, _, self.Tr_velo_to_cam = load_kitti_calib(calib_path)
        else:
            self.Tr_velo_to_cam = DEFAULT_TR_VELO_TO_CAM.copy()

    def project_points(self, points_velo):
        uv, z_cam = project_lidar_to_camera(points_velo, self.R0_rect, self.Tr_velo_to_cam, self.K)
        H_img = int(2 * self.K[1, 2]) if self.K[1, 2] > 0 else 375
        W_img = int(2 * self.K[0, 2]) if self.K[0, 2] > 0 else 1242
        mask = (uv[:, 0] >= 0) & (uv[:, 0] < W_img) & (uv[:, 1] >= 0) & (uv[:, 1] < H_img) & (z_cam > 0)
        return uv, z_cam, mask

    def fuse_bbox_depth(self, stereo_depth_map, detections, points_velo):
        uv, z_cam, mask = self.project_points(points_velo)
        uv_valid, z_valid = uv[mask], z_cam[mask]
        H, W = stereo_depth_map.shape[:2]
        results = []

        for det in detections:
            bbox, score, class_id, obj_id = det
            lidar_d, lidar_n = get_lidar_depth_in_bbox(uv_valid, z_valid, bbox, H, W)

            x1, y1, x2, y2 = [int(c) for c in bbox]
            x1, x2 = max(0, min(x1, W - 1)), max(0, min(x2, W - 1))
            y1, y2 = max(0, min(y1, H - 1)), max(0, min(y2, H - 1))
            region = stereo_depth_map[y1:y2, x1:x2]
            valid_region = region[region > 0]
            stereo_d = float(np.median(valid_region)) if valid_region.size > 0 else 0.0

            fused_d, source = fuse_depth_in_bbox(
                stereo_d, lidar_d, lidar_n,
                stereo_weight=self.stereo_weight,
                min_lidar_points=self.min_lidar_points,
                max_fuse_dist=self.max_fuse_dist
            )

            results.append({
                'depth_value': fused_d, 'source': source, 'lidar_points': lidar_n,
                'stereo_depth': stereo_d, 'lidar_depth': lidar_d,
            })
        return results