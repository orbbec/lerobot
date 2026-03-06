<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/docs/source/CODE_OF_CONDUCT.zh-CN.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeRobot** aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry so that everyone can contribute to and benefit from shared datasets and pretrained models.

🤗 A hardware-agnostic, Python-native interface that standardizes control across diverse platforms, from low-cost arms (SO-100) to humanoids.

🤗 A standardized, scalable LeRobotDataset format (Parquet + MP4 or images) hosted on the Hugging Face Hub, enabling efficient storage, streaming and visualization of massive robotic datasets.

🤗 State-of-the-art policies that have been shown to transfer to the real-world ready for training and deployment.

🤗 Comprehensive support for the open-source ecosystem to democratize physical AI.

## Quick Start

LeRobot can be installed directly from PyPI.

```bash
pip install lerobot
lerobot-info
```

> [!IMPORTANT]
> For detailed installation guide, please see the [Installation Documentation](https://huggingface.co/docs/lerobot/installation).

## Robots & Control

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot provides a unified `Robot` class interface that decouples control logic from hardware specifics. It supports a wide range of robots and teleoperation devices.

```python
from lerobot.robots.myrobot import MyRobot

# Connect to a robot
robot = MyRobot(config=...)
robot.connect()

# Read observation and send action
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

**Supported Hardware:** SO100, LeKiwi, Koch, HopeJR, OMX, EarthRover, Reachy2, Gamepads, Keyboards, Phones, OpenARM, Unitree G1.

While these devices are natively integrated into the LeRobot codebase, the library is designed to be extensible. You can easily implement the Robot interface to utilize LeRobot's data collection, training, and visualization tools for your own custom robot.

For detailed hardware setup guides, see the [Hardware Documentation](https://huggingface.co/docs/lerobot/integrate_hardware).

## LeRobot Dataset

To solve the data fragmentation problem in robotics, we utilize the **LeRobotDataset** format.

- **Structure:** Synchronized MP4 videos (or images) for vision and Parquet files for state/action data.
- **HF Hub Integration:** Explore thousands of robotics datasets on the [Hugging Face Hub](https://huggingface.co/lerobot).
- **Tools:** Seamlessly delete episodes, split by indices/fractions, add/remove features, and merge multiple datasets.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset from the Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access data (automatically handles video decoding)
episode_index=0
print(f"{dataset[episode_index]['action'].shape=}\n")
```

Learn more about it in the [LeRobotDataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## SoTA Models

LeRobot implements state-of-the-art policies in pure PyTorch, covering Imitation Learning, Reinforcement Learning, and Vision-Language-Action (VLA) models, with more coming soon. It also provides you with the tools to instrument and inspect your training process.

<p align="center">
  <img alt="Gr00t Architecture" src="./media/readme/VLA_architecture.jpg" width="640px">
</p>

Training a policy is as simple as running a script configuration:

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

| Category                   | Models                                                                                                                                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Imitation Learning**     | [ACT](./docs/source/policy_act_README.md), [Diffusion](./docs/source/policy_diffusion_README.md), [VQ-BeT](./docs/source/policy_vqbet_README.md)                                                             |
| **Reinforcement Learning** | [HIL-SERL](./docs/source/hilserl.mdx), [TDMPC](./docs/source/policy_tdmpc_README.md) & QC-FQL (coming soon)                                                                                                  |
| **VLAs Models**            | [Pi0Fast](./docs/source/pi0fast.mdx), [Pi0.5](./docs/source/pi05.mdx), [GR00T N1.5](./docs/source/policy_groot_README.md), [SmolVLA](./docs/source/policy_smolvla_README.md), [XVLA](./docs/source/xvla.mdx) |

Similarly to the hardware, you can easily implement your own policy & leverage LeRobot's data collection, training, and visualization tools, and share your model to the HF Hub

For detailed policy setup guides, see the [Policy Documentation](https://huggingface.co/docs/lerobot/bring_your_own_policies).

## Inference & Evaluation

Evaluate your policies in simulation or on real hardware using the unified evaluation script. LeRobot supports standard benchmarks like **LIBERO**, **MetaWorld** and more to come.

```bash
# Evaluate a policy on the LIBERO benchmark
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

Learn how to implement your own simulation environment or benchmark and distribute it from the HF Hub by following the [EnvHub Documentation](https://huggingface.co/docs/lerobot/envhub)

## Orbbec RGBD Support

This fork extends LeRobot with native support for **Orbbec depth cameras** and a **4-channel RGBD variant of the ACT policy**, validated on the SO-101 robotic arm.

> Special thanks to [@ImpurestTadpole](https://github.com/ImpurestTadpole) for the open-source RGBD implementation.
> Reference: [RGBD_IMPLEMENTATION_GUIDE.md](https://github.com/ImpurestTadpole/lerobot/blob/main/RGBD_IMPLEMENTATION_GUIDE.md)

### What's Added

| Component            | Description                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------------- |
| `OrbbecCamera`       | Camera driver for Orbbec depth cameras via `pyorbbecsdk2`, supporting synchronized color + depth capture       |
| `OrbbecCameraConfig` | Configuration with `use_depth`, `align_depth`, `d2c_mode` (hardware/software D2C alignment)                    |
| RGBD ACT             | ACT policy extended to accept 4-channel (RGB + Depth) input; ResNet `conv1` is modified from 3-ch to 4-ch      |
| SO-101 + Orbbec      | End-to-end validation of data collection, training, and inference on the SO-101 arm with an Orbbec RGBD camera |

### Installation

Install the Orbbec SDK Python bindings in addition to the standard LeRobot dependencies:

```bash
pip install pyorbbecsdk2==2.0.18
```

> [!WARNING]
> **Linux — first-time setup:** Before using an Orbbec camera for the first time, install the udev rules so the device is accessible. Run once after installing the package:
>
> ```bash
> SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
> sudo chmod +x "$SITE_PACKAGES/pyorbbecsdk/shared/install_udev_rules.sh"
> sudo "$SITE_PACKAGES/pyorbbecsdk/shared/install_udev_rules.sh"
> sudo udevadm control --reload-rules && sudo udevadm trigger
> ```
>
> For more details, see the [pyorbbecsdk installation guide](https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html#verify-the-installation).

### Discover Connected Orbbec Cameras

Run the following command to list all connected Orbbec devices:

```bash
lerobot-find-cameras orbbec
```

Example output:

```
--- Detected Cameras ---
Camera #0:
  Type: Orbbec
  Index: 0
  Name: Orbbec Gemini 336L
  Id: CPCG85300038
  Connection type: USB3.2
  Has color sensor: True
  Has depth sensor: True
  Default color profile: {'width': 1280, 'height': 720, 'fps': 30, 'format': 'OBFormat.MJPG'}
  Default depth profile: {'width': 848, 'height': 480, 'fps': 30, 'format': 'OBFormat.Y16'}
--------------------
Camera #1:
  Type: Orbbec
  Index: 1
  Name: Orbbec Gemini 336
  Id: CP99853000V8
  Connection type: USB3.2
  Has color sensor: True
  Has depth sensor: True
  Default color profile: {'width': 1280, 'height': 720, 'fps': 30, 'format': 'OBFormat.MJPG'}
  Default depth profile: {'width': 848, 'height': 480, 'fps': 30, 'format': 'OBFormat.Y16'}
--------------------
```

The `Id` field (serial number) is what you pass as `index_or_serial_number` in the camera config. Using the serial number rather than `Index` is recommended — the index may change after a reboot or USB replug, while the serial number is stable.

### Collect a Dataset

Use `lerobot-record` to teleoperate the SO-101 and record an RGBD dataset. The example below mounts three cameras: one OpenCV wrist camera and two Orbbec cameras (one colour-only, one with software-aligned depth):

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/so101_follower \
  --robot.id=so101_follower \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": 0,
      "width": 640,
      "height": 480,
      "fps": 30
    },
    "left": {
      "type": "orbbec",
      "index_or_serial_number": "CPCG85300038",
      "width": 640,
      "height": 480,
      "fps": 30
    },
    "head": {
      "type": "orbbec",
      "index_or_serial_number": "CP99853000V8",
      "width": 640,
      "height": 480,
      "fps": 30,
      "use_depth": true,
      "align_depth": true,
      "d2c_mode": "software"
    }
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101_leader \
  --teleop.id=so101_leader \
  --dataset.repo_id=<your-rgbd-dataset> \
  --dataset.single_task="pick up the cube and place it in the box" \
  --dataset.num_episodes=50 \
  --dataset.fps=30 \
  --display_data=true
```

Key points:

- `wrist` — standard USB webcam via `OpenCVCamera`
- `left` — Orbbec Gemini 336L (`Id: CPCG85300038`), colour only (no depth)
- `head` — Orbbec Gemini 336 (`Id: CP99853000V8`), depth enabled with software D2C alignment
- Depth frames are stored under the key `observation.images.head_depth` in the dataset

### Train ACT with RGBD Input

The ACT policy automatically detects depth features by looking for keys matching `<camera_key>_depth` in the dataset. When depth features are present, the ResNet backbone's first convolution layer is expanded to accept a 4-channel (RGB + D) input. Depth values are normalised to `[0, 1]` by dividing by `depth_max_range` before being concatenated with the RGB channels. It is recommended to set `depth_max_range` to match the actual working distance of your task — for SO-101 tabletop grasping, 1 m is a good default (values beyond 1 m are clamped to 1.0).

```bash
lerobot-train \
  --policy=act \
  --policy.depth_max_range=1.0 \
  --dataset.repo_id=<your-rgbd-dataset>
```

For a full step-by-step guide covering data collection, dataset creation, training, and deployment, see the [RGBD Implementation Guide](https://github.com/ImpurestTadpole/lerobot/blob/main/RGBD_IMPLEMENTATION_GUIDE.md).

## Resources

- **[Documentation](https://huggingface.co/docs/lerobot/index):** The complete guide to tutorials & API.
- **[Chinese Tutorials: LeRobot+SO-ARM101中文教程-同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)** Detailed doc for assembling, teleoperate, dataset, train, deploy. Verified by Seed Studio and 5 global hackathon players.
- **[Discord](https://discord.gg/q8Dzzpym3f):** Join the `LeRobot` server to discuss with the community.
- **[X](https://x.com/LeRobotHF):** Follow us on X to stay up-to-date with the latest developments.
- **[Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial):** A free, hands-on course to learn robot learning using LeRobot.

## Citation

If you use LeRobot in your research, please cite:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

## Contribute

We welcome contributions from everyone in the community! To get started, please read our [CONTRIBUTING.md](./CONTRIBUTING.md) guide. Whether you're adding a new feature, improving documentation, or fixing a bug, your help and feedback are invaluable. We're incredibly excited about the future of open-source robotics and can't wait to work with you on what's next—thank you for your support!

<p align="center">
  <img alt="SO101 Video" src="./media/readme/so100_video.webp" width="640px">
</p>

<div align="center">
<sub>Built by the <a href="https://huggingface.co/lerobot">LeRobot</a> team at <a href="https://huggingface.co">Hugging Face</a> with ❤️</sub>
</div>
