# Data Collection Scripts

This folder contains Python scripts for collecting various types of robotic simulation data using the Robosuite framework. Each script is designed for specific data collection scenarios and camera configurations.

## Scripts Overview

### Core Data Collection Scripts

#### `orbit_wiping_camera.py`
- **Purpose**: Collects eye-in-hand camera data during a spiral orbit motion around a cube in the Lift environment
- **Robot**: Panda robot with wiping gripper
- **Motion**: Spiral trajectory with 3 revolutions around a cube
- **Cameras**: robot0_eye_in_hand, agentview, frontview, sideview, birdview
- **Output**: Creates `robosuite_lift_spiral_dataset/` with images and transforms.json
- **Parameters**: 360 waypoints, 0.25m radius, 512x512 images

#### `baxter_lift_spiral.py`
- **Purpose**: Similar to orbit_wiping_camera.py but uses Baxter robot for larger workspace
- **Robot**: Baxter robot (dual-arm)
- **Motion**: Spiral orbit motion around a cube
- **Cameras**: robot0_eye_in_hand, agentview, frontview, sideview, birdview
- **Output**: Creates `robosuite_baxter_lift_spiral_dataset/` with images and transforms.json
- **Parameters**: 360 waypoints, 0.6m radius (larger for Baxter), 512x512 images

#### `orbit_camera.py`
- **Purpose**: Collects hemispherical camera data for eye-in-hand camera
- **Robot**: Panda robot
- **Motion**: Hemisphere scan pattern with multiple vertical levels
- **Cameras**: robot0_eye_in_hand
- **Output**: Creates `robosuite_hemisphere_dataset/` with images and transforms.json
- **Parameters**: 5 vertical levels, 72 horizontal steps (360 total), 0.4m orbit radius

#### `panda_wiping_sim.py`
- **Purpose**: Interactive simulation for Panda robot with wiping gripper
- **Robot**: Panda robot with WipingGripper
- **Environment**: Wipe task environment
- **Cameras**: robot0_eye_in_hand, agentview, frontview, sideview
- **Output**: Saves images to `panda_wiping_images/` directory
- **Features**: Interactive control with keyboard input (q=quit, s=save, r=reset)

### Multi-Camera Scripts

#### `run_multicam.py`
- **Purpose**: Demonstrates multi-camera setup with 4 cameras arranged equidistantly
- **Robot**: Panda robot
- **Environment**: Lift task
- **Cameras**: agentview, frontview, sideview, birdview (arranged in circle)
- **Features**: Real-time visualization of all camera views

#### `run_task.py`
- **Purpose**: Interactive task execution with multi-camera setup
- **Robot**: Panda robot
- **Environment**: Lift task with custom multi-camera configuration
- **Cameras**: robot0_eye_in_hand, agentview, frontview, sideview, birdview
- **Features**: Interactive control and real-time camera switching

#### `static_image.py`
- **Purpose**: Captures static images from multiple camera angles
- **Robot**: Panda robot
- **Environment**: Lift task
- **Cameras**: robot0_eye_in_hand, agentview, frontview, sideview, birdview
- **Output**: Saves individual camera images to `camera_images/` directory

### Utility Scripts

#### `create_video.py`
- **Purpose**: Creates MP4 videos from collected image sequences
- **Input**: Directory of PNG images (e.g., from dataset collection)
- **Output**: MP4 video file
- **Features**: Natural sorting of frame files, configurable frame rate (30 FPS default)
- **Usage**: Modify IMAGE_DIR and OUTPUT_VIDEO_FILE variables

## Environment Setup

All scripts require:
- Robosuite framework
- OpenCV (cv2)
- NumPy
- JSON support
- Natural sorting library (natsort) for video creation

## Camera Configurations

### Standard Cameras
- **robot0_eye_in_hand**: Eye-in-hand camera mounted on robot end-effector
- **agentview**: Third-person view of the robot and environment

### Custom Overhead Cameras
- **frontview**: Positioned in front, looking back and down
- **sideview**: Positioned to the side, angled view
- **birdview**: High overhead view looking straight down

## Output Formats

### Dataset Structure
```
dataset_name/
├── images/
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
└── transforms.json
```

### Transforms.json Format
Contains camera poses and intrinsic parameters for each frame:
- Camera position and orientation
- Focal length and field of view
- Image dimensions
- Frame timestamps

## Usage Examples

### Collect Spiral Dataset
```bash
python orbit_wiping_camera.py
```

### Create Video from Images
```bash
python create_video.py
```

### Interactive Simulation
```bash
python panda_wiping_sim.py
```

## Notes

- All scripts use offscreen rendering for data collection
- Image dimensions are typically 512x512 or 256x256
- Control frequency is set to 20 Hz for smooth motion
- Scripts automatically create output directories
- Existing datasets are overwritten when re-running scripts
