# 基于 Fast-Foundation Stereo 和 YOLO v11 的实时3D目标跟踪系统

这是一个实时3D物体检测系统，将用于物体检测的YOLOv11与用于深度估计的Fast-Foundation Stereo相结合，从而生成3D边界框并实现BEV可视化。

# 项目结构

```
├── detect_and_track/          # 主要代码
│   ├── run_stereo.py          # 主入口
│   ├── stereo_depth_model.py  # Fast-FoundationStereo 深度封装
│   ├── detection_model.py     # YOLOv11 检测器
│   ├── bbox3d_utils.py        # 3D 框估计 + 可视化
│   ├── load_camera_params.py  # 相机参数加载
│   ├── lidar_fusion.py        # 点云、视觉深度融合
│   └── test_single.py         # 单帧诊断脚本
├── core/                      # Fast-FoundationStereo 核心
│   ├── foundation_stereo.py   # 模型定义
│   ├── extractor.py           # 特征提取
│   ├── geometry.py            # 几何编码
│   ├── submodule.py           # 子模块
│   ├── update.py              # GRU 更新
│   └── utils/
├── Utils.py                   # 工具函数
├── requirements.txt           # Python 依赖
└── .gitignore
```

# 安装环境

```
conda env create -f requirements.yml
conda activate fast_foundation_stereo
```

# 模型权重

从 [Google Drive](https://drive.google.com/drive/folders/1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap?usp=drive_link)下载模型权重，并放在 `weights/` (e.g. `./weights/23-36-37`). 下表对比了预训练模型家族中不同尺寸的若干代表性模型之间的差异。这些模型按运行速度从慢到快排序，准确率依次递减；运行时间是在 GPU 3090 上、图像尺寸为 640x480 的条件下测得的。

为了权衡速度和精度，可以尝试:

1) 不同的权重.

2) 调整配置设置 (see explanations in the "Run demo" section below).

| Checkpoint     | valid_iters | Runtime-Pytorch (ms) | Runtime-TRT (ms) | Peak Memory (MB) |
|---------------|-------------|-------------|-----------------|-----------------|
| `23-36-37`    | 8           | 49.4        | 23.4            | 653             |
| `23-36-37`    | 4           | 41.1        | 18.4            | 653             |
| `20-26-39`    | 8           | 43.6        | 19.4            | 651             |
| `20-26-39`    | 4           | 37.5        | 16.4            | 651             |
| `20-30-48`    | 8           | 38.4        | 16.6            | 646             |
| `20-30-48`    | 4           | 29.3        | 14.0            | 646             |

```
weights/
└── 20-30-48/
    └── model_best_bp2_serialize.pth
```

# 运行

```
python detect_and_track/run_stereo.py --ckpt_dir weights/23-36-37/model_best_bp2_serialize.pth --left_dir demo_data/left --right_dir demo_data/right  --lidar_dir /velodyne_points/data --calib_path /path/to/calib_cam_to_cam.txt --calib_velo_path /path/to/calib_velo_to_cam.txt --output_path output/output_faststereo.mp4  --scale 1 --valid_iters 8 --max_frame 192 --z_far 100
```

如果传参`--lidar_dir`,则深度图为雷达点云深度和双目视觉深度融合获取，如果不传则退化为纯双目视觉深度。

| Flag                        | Meaning                                                                |
|-----------------------------|------------------------------------------------------------------------|
| `--ckpt_dir`               | 模型权重路径                                 |
| `--left_dir`               | 左视图路径                                            |
| `--right_dir`              | 右视图路径                                          |
| `--output_path `           | 输出文件路径                                    |
| `--lidar_dir `           | 雷达点云路径                                    |
| `--calib_path `           | 双目相机内参和校正矩阵路径                                    |
| `--calib_velo_path `           | LiDAR到相机外参路径                                    |
| `--scale`                   | 图像缩放比例系数                                                   |
| `--valid_iters`             | 前向传播过程中的优化更新次数                       |
| `--max_frame`                | 最大处理帧数 |
| `--zfar`                    | BEV可视化的最大深度                                |

有关参数的完整列表，请参阅 `detect_and_track/run_stereo.py`。

注：
- 左右眼输入数据需要是序列图像，且应经过校正，无畸变，左右图像之间的极线应呈水平方向。可以使用[KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php)官方数据
- 请勿调换左右图像的位置。左图必须确实来自左侧摄像头（图像中的物体应向右偏移）。
- 我们建议使用无损压缩的 PNG 文件。
- 对于高分辨率图像（>1000px），您可以：(1) 使用 `--hiera 1` 启用分层推理，以获得全分辨率深度图，但速度较慢；或者 (2) 使用较小的缩放比例，例如 `--scale 0.5`，以获得缩小分辨率但速度更快的深度图。
若需加快推理速度，可通过 `--scale 0.5` 等参数降低输入图像分辨率，并减少精化迭代次数，例如使用 `--valid_iters 4`.

## 演示视频

[演示视频](https://www.bilibili.com/video/BV1eNTy6AEcD/)

| 模型     | 推理速度(3080 10G)|输入大小|1242x375||
|----------|-|-|-|----------------------------|
|      | total | Detect | Depth | Avg FPS |
| `20-30-48` (valid_iters=8)camara_only  | 192ms|12ms|172ms|5.3 |
| `20-30-48` (valid_iters=8)lidar_fusion  | 198ms|11ms|181ms|5.1 |

其余配置选项和[FoundationStereo Detector](https://github.com/Lao-G-G/FoundationStereo-based-YOLO-3D-detector/tree/main)一致。

## 工作流

```
左目图像 ──→ YOLOv11 2D 检测 ──→ 2D 边界框 + 类别 + 跟踪 ID
                                        ↓
                                  雷达点云获取检测框深度
                                        ↓
左目+右目 ──→ FoundationStereo ──→ 全图像素级 metric 深度图，雷达深度微调
              ↑                         ↓
        2D 框 + 深度值 + 相机内参 → 反向投影 → 3D 边界框 (x, y, z, 尺寸, 朝向)
                                           ↓
                                      可视化：3D 立体框 + BEV 俯视图
```

1. **目标检测**：YOLOv11 检测左眼视图中的目标并提供 2D 边界框
2. **深度估计**：Foundation Stereo 根据左右眼视差为整个画面生成深度图，用雷达点云数据获取检测框深度，微调检测框的深度图
3. **3D 边界框估计**：根据检测框的深度估计以及左右眼视差生成 3D 边界框
4. **可视化**：渲染 3D 边界框和俯视图，以更好地理解空间关系

## 诊断

如果运行报错可以使用`test_single.py`查看具体是哪个模块报错。

# 致谢

[YOLOv11 by Ultralytics](https://github.com/ultralytics/ultralytics)

[Depth Anything v2 by TikTok](https://github.com/DepthAnything/Depth-Anything-V2)

[Fast-Foundation Stereo by Nvidia](https://github.com/NVlabs/Fast-FoundationStereo)
