import robosuite as suite
import numpy as np
import cv2
import os
import json
import shutil

# --- Setup ---
OUTPUT_DIR = "robosuite_hemisphere_dataset"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Hemisphere Parameters ---
ORBIT_RADIUS = 0.4
VERTICAL_LEVELS = 5
HORIZONTAL_STEPS = 72
TOTAL_STEPS = VERTICAL_LEVELS * HORIZONTAL_STEPS

# --- Environment Definition ---
CAMERA_NAME = "robot0_eye_in_hand"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    control_freq=20,
    camera_names=[CAMERA_NAME],
    camera_heights=IMAGE_HEIGHT,
    camera_widths=IMAGE_WIDTH,
    horizon=10000,  # Very large horizon for data collection
    ignore_done=True,  # Ignore episode termination
)

# --- Data Collection Logic ---
print("Starting hemispherical eye-in-hand data collection...")
obs = env.reset()
frames_data = []

cam_id = env.sim.model.camera_name2id(CAMERA_NAME)
fovy = env.sim.model.cam_fovy[cam_id]
fy = IMAGE_HEIGHT / (2 * np.tan(np.deg2rad(fovy) / 2))
fx = IMAGE_WIDTH / (2 * np.tan(np.deg2rad(fovy) / 2))

frame_count = 0

# --- Nested Loop for Hemispherical Motion ---
for i in range(VERTICAL_LEVELS):
    polar_angle_rad = np.deg2rad(10 + i * (70 / (VERTICAL_LEVELS - 1)))
    
    for j in range(HORIZONTAL_STEPS):
        azimuth_angle_rad = (j / HORIZONTAL_STEPS) * 2 * np.pi

        # 1. Calculate target gripper pose
        cube_pos = obs['cube_pos']
        dx = ORBIT_RADIUS * np.sin(polar_angle_rad) * np.cos(azimuth_angle_rad)
        dy = ORBIT_RADIUS * np.sin(polar_angle_rad) * np.sin(azimuth_angle_rad)
        dz = ORBIT_RADIUS * np.cos(polar_angle_rad)
        target_pos = cube_pos + np.array([dx, dy, dz])
        
        eef_pos = obs['robot0_eef_pos']
        action = np.zeros(env.action_dim)
        action[:3] = (target_pos - eef_pos) * 5
        action[6] = -1

        # 2. Step the simulation
        for _ in range(10):
             obs, reward, done, info = env.step(action)
             # No need to check for episode termination since we set ignore_done=True
        
        # 3. Save the frame data
        img = obs[f"{CAMERA_NAME}_image"]
        
        current_cam_pos = env.sim.data.cam_xpos[cam_id]
        current_cam_rot = env.sim.data.cam_xmat[cam_id].reshape(3, 3)
        
        c2w_matrix = np.eye(4)
        c2w_matrix[:3, :3] = current_cam_rot
        c2w_matrix[:3, 3] = current_cam_pos
        
        correction = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        c2w_matrix = c2w_matrix @ correction
        
        image_filename = f"frame_{frame_count:04d}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        
        img_bgr = cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, img_bgr)
        
        frames_data.append({"file_path": f"images/{image_filename}", "transform_matrix": c2w_matrix.tolist()})

        # Live Feed
        cv2.imshow("Live Camera Feed", img_bgr)
        cv2.waitKey(1)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Collected frame {frame_count}/{TOTAL_STEPS}")

# --- Create the transforms.json file ---
output_json = {
    "camera_angle_x": np.arctan(IMAGE_WIDTH / (2 * fx)),
    "fl_x": fx, "fl_y": fy, "cx": IMAGE_WIDTH / 2, "cy": IMAGE_HEIGHT / 2,
    "w": IMAGE_WIDTH, "h": IMAGE_HEIGHT, "frames": frames_data
}
with open(os.path.join(OUTPUT_DIR, "transforms.json"), "w") as f:
    json.dump(output_json, f, indent=4)

# Cleanup
cv2.destroyAllWindows()
env.close()
print(f"\nHemispherical data collection complete! Dataset saved in '{OUTPUT_DIR}'")