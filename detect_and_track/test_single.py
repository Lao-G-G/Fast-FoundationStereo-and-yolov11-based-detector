#!/usr/bin/env python3
"""Single image diagnostic test for YOLO + Fast-FoundationStereo pipeline."""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys, time, traceback, cv2
import numpy as np
import torch

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(code_dir, '..'))
sys.path.insert(0, os.path.join(code_dir, '..', 'core'))

# ========== CONFIG ==========
CKPT = r"C:\Users\GGG\Downloads\20-30-48-20260705T074058Z-3-001\20-30-48\model_best_bp2_serialize.pth"
LEFT = None   # Will auto-find from KITTI default or specify path here
RIGHT = None
BASELINE = 0.54
# ============================

print("=" * 60)
print("DIAGNOSTIC TEST")
print("=" * 60)

# Step 1: Check imports
print("\n[1] Checking imports...")
try:
    from detection_model import ObjectDetector
    print("  detection_model ✓")
except Exception as e:
    print(f"  detection_model ✗: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from stereo_depth_model import StereoDepthEstimator
    print("  stereo_depth_model ✓")
except Exception as e:
    print(f"  stereo_depth_model ✗: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Check weights file
print(f"\n[2] Checking weights: {CKPT}")
if os.path.exists(CKPT):
    print(f"  ✓ Exists ({os.path.getsize(CKPT)/1024/1024:.1f} MB)")
else:
    print(f"  ✗ NOT FOUND")
    sys.exit(1)

# Step 3: Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"\n[3] Device: {device}")

# Step 4: Auto-find test images
if LEFT is None:
    default_dir = r"D:\PyCharm_2025.1.1.1\project\YOLOStereo3D-master\dataset\2011_09_26_drive_0017_sync\2011_09_26\2011_09_26_drive_0017_sync"
    left_dir = os.path.join(default_dir, "image_02", "data")
    right_dir = os.path.join(default_dir, "image_03", "data")
    if os.path.isdir(left_dir):
        import glob
        lefts = sorted(glob.glob(os.path.join(left_dir, "*.png")))
        rights = sorted(glob.glob(os.path.join(right_dir, "*.png")))
        if lefts and rights:
            LEFT = lefts[0]
            RIGHT = rights[0]

print(f"[4] Test images:")
print(f"  Left:  {LEFT}")
print(f"  Right: {RIGHT}")
if not LEFT or not os.path.exists(LEFT):
    print("  ✗ Left image not found!")
    sys.exit(1)
if not RIGHT or not os.path.exists(RIGHT):
    print("  ✗ Right image not found!")
    sys.exit(1)

# Step 5: Init YOLO
print("\n[5] Initializing YOLO detector...")
try:
    detector = ObjectDetector(model_size='nano', conf_thres=0.25, iou_thres=0.45, device=device)
    print("  ✓ YOLO initialized")
except:
    print("  Trying CPU fallback...")
    detector = ObjectDetector(model_size='nano', conf_thres=0.25, iou_thres=0.45, device='cpu')
    print("  ✓ YOLO initialized (CPU)")

# Step 6: Init Fast-FoundationStereo
print("\n[6] Initializing Fast-FoundationStereo...")
try:
    depth_estimator = StereoDepthEstimator(
        ckpt_dir=CKPT,
        baseline=BASELINE,
        device=str(device),
        scale=1.0,
        valid_iters=8
    )
    print("  ✓ Model loaded")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 7: Read images
print("\n[7] Reading images...")
left_img = cv2.imread(LEFT)
right_img = cv2.imread(RIGHT)
print(f"  Left shape: {left_img.shape}, dtype: {left_img.dtype}")
print(f"  Right shape: {right_img.shape}, dtype: {right_img.dtype}")

# Step 8: Run YOLO
print("\n[8] Running YOLO detection...")
t0 = time.time()
det_frame, detections = detector.detect(left_img, track=True)
t1 = time.time()
print(f"  Detections: {len(detections)} objects found ({1000*(t1-t0):.0f}ms)")
for d in detections[:5]:
    bbox, score, cls_id, obj_id = d
    cls_name = detector.get_class_names()[cls_id]
    print(f"    [{obj_id}] {cls_name} score={score:.2f} bbox={[int(x) for x in bbox]}")

# Step 9: Run depth estimation
print("\n[9] Running Fast-FoundationStereo depth estimation...")
print("    (First run: CUDA JIT compilation, may take 10-30s)")
t0 = time.time()
try:
    depth_meters = depth_estimator.estimate_depth(left_img, right_img)
    t1 = time.time()
    print(f"  ✓ Depth shape: {depth_meters.shape}, dtype: {depth_meters.dtype}")
    valid = depth_meters > 0
    print(f"  Valid pixels: {valid.sum()} / {depth_meters.size} ({100*valid.sum()/depth_meters.size:.1f}%)")
    if valid.sum() > 0:
        print(f"  Depth range: {depth_meters[valid].min():.2f}m - {depth_meters[valid].max():.2f}m")
    else:
        print("  ✗ ALL ZEROS! No valid depth values.")
    print(f"  Time: {1000*(t1-t0):.0f}ms")

    # Colorize
    depth_colored = depth_estimator.colorize_depth(depth_meters)
    print(f"  Colorized shape: {depth_colored.shape}")

    # Save debug images
    cv2.imwrite("debug_left.png", left_img)
    cv2.imwrite("debug_right.png", right_img)
    cv2.imwrite("debug_depth.png", depth_colored)
    print("  Debug images saved: debug_left.png, debug_right.png, debug_depth.png")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    traceback.print_exc()
    depth_meters = np.zeros(left_img.shape[:2], dtype=np.float32)
    depth_colored = np.zeros((*left_img.shape[:2], 3), dtype=np.uint8)

# Step 10: Build 3D boxes
print("\n[10] Building 3D boxes...")
from bbox3d_utils import BBox3DEstimator
bbox3d_estimator = BBox3DEstimator(
    camera_matrix=depth_estimator.K,
    projection_matrix=np.hstack([depth_estimator.K, np.zeros((3, 1))]),
    use_metric_depth=True
)

boxes_3d = []
for detection in detections:
    bbox, score, class_id, obj_id = detection
    class_name = detector.get_class_names()[class_id]
    if class_name.lower() in ['person', 'cat', 'dog']:
        cx = int((bbox[0]+bbox[2])/2); cy = int((bbox[1]+bbox[3])/2)
        dv = depth_estimator.get_depth_at_point(depth_meters, cx, cy)
    else:
        dv = depth_estimator.get_depth_in_region(depth_meters, bbox, method='median')
    if dv <= 0:
        print(f"  Skip {class_name}: depth={dv:.2f}")
        continue
    box_3d = bbox3d_estimator.estimate_3d_box(bbox, dv, class_name, object_id=obj_id)
    box_3d['depth_value_metric'] = dv
    boxes_3d.append(box_3d)
    print(f"  {class_name}: depth={dv:.2f}m, location={box_3d['location'].round(2)}")

print(f"\nTotal 3D boxes: {len(boxes_3d)}")

if len(boxes_3d) > 0:
    # Draw
    result = left_img.copy()
    for b in boxes_3d:
        result = bbox3d_estimator.draw_box_3d(result, b)
    cv2.imwrite("debug_result.png", result)
    print("  Debug result saved: debug_result.png")
    cv2.imshow("Result", result)
    cv2.imshow("Depth", depth_colored)
    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("\nNo 3D boxes generated. Check detections and depth values.")
    cv2.imshow("Left", left_img)
    cv2.imshow("Depth (all zeros?)", depth_colored)
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)