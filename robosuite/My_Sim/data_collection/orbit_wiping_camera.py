import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
import numpy as np
import cv2
import os
import json
import shutil

# --- Setup ---
OUTPUT_DIR = "robosuite_lift_spiral_dataset"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
AGENTVIEW_DIR = os.path.join(OUTPUT_DIR, "agentview_images")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(AGENTVIEW_DIR, exist_ok=True)

# --- Spiral Parameters ---
SPHERE_RADIUS = 0.25  # Smaller radius to stay within workspace
SPHERE_CENTER = np.array([0.0, 0.0, 0.9])  # Center around the cube
NUM_POINTS = 360  # Total number of waypoints
NUM_REVOLUTIONS = 3  # Fewer revolutions for safer trajectory

# --- Environment Definition ---
CAMERA_NAME = "robot0_eye_in_hand"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# Custom Lift environment class with additional cameras
class LiftOrbitEnv(Lift):
    def _setup_cameras(self):
        """Set up cameras for visualization"""
        super()._setup_cameras()
        
        # Add custom cameras for better viewing
        self.sim.model.camera_add(
            "frontview",
            pos=[0.6, 0, 1.2],
            quat=[0.383, 0, 0.924, 0],
        )
        
        self.sim.model.camera_add(
            "sideview", 
            pos=[-0.3, 0.5, 1.2],
            quat=[0.191, -0.800, 0.462, 0.332],
        )
        
        self.sim.model.camera_add(
            "birdview",
            pos=[0.0, 0.0, 1.8],  # High above looking down
            quat=[0.707, 0, 0, 0.707],  # Looking straight down
        )

env = LiftOrbitEnv(
    robots="Panda",
    gripper_types="WipingGripper",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    control_freq=20,
    camera_names=[CAMERA_NAME, "agentview", "frontview", "sideview"],
    camera_heights=IMAGE_HEIGHT,
    camera_widths=IMAGE_WIDTH,
    horizon=10000,  # Very large horizon for data collection
    ignore_done=True,  # Ignore episode termination
)

def is_in_workspace(pos, center, max_radius=0.6):
    """
    Check if a position is within the robot's workspace.
    
    Args:
        pos (np.ndarray): Position to check
        center (np.ndarray): Workspace center
        max_radius (float): Maximum reachable radius
    
    Returns:
        bool: True if position is reachable
    """
    distance = np.linalg.norm(pos - center)
    return distance <= max_radius

def is_cube_visible(eef_pos, cube_pos, camera_fov_deg=75, min_distance=0.075, max_distance=1.0):
    """
    Check if the cube is visible from the camera position.
    
    Args:
        eef_pos (np.ndarray): End-effector position
        cube_pos (np.ndarray): Cube position
        camera_fov_deg (float): Camera field of view in degrees
        min_distance (float): Minimum distance for visibility
        max_distance (float): Maximum distance for visibility
    
    Returns:
        bool: True if cube is visible
    """
    # Calculate distance to cube
    distance = np.linalg.norm(cube_pos - eef_pos)
    
    # Check distance constraints
    if distance < min_distance or distance > max_distance:
        return False
    
    # Calculate angle between camera Z-axis and direction to cube
    # Camera Z-axis points down [0, 0, -1] in robot frame
    camera_z = np.array([0, 0, -1])
    direction_to_cube = cube_pos - eef_pos
    direction_to_cube = direction_to_cube / np.linalg.norm(direction_to_cube)
    
    # Calculate angle between vectors
    cos_angle = np.dot(camera_z, direction_to_cube)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = np.rad2deg(angle_rad)
    
    # Check if angle is within field of view
    return angle_deg <= camera_fov_deg / 2

def generate_spherical_spiral_waypoints(radius=0.5, center=np.array([0, 0, 0]), cube_pos=np.array([0, 0, 0.9]), num_points=200, num_revolutions=5):
    """
    Generates waypoints for a spiral path on the upper hemisphere of a sphere.
    Only includes waypoints where the cube is visible in the camera's field of view.

    Args:
        radius (float): The radius of the sphere.
        center (np.ndarray): A 3-element numpy array for the [x, y, z] center of the sphere.
        cube_pos (np.ndarray): A 3-element numpy array for the [x, y, z] position of the cube.
        num_points (int): The number of waypoints to generate for the path.
        num_revolutions (float): The total number of revolutions the spiral makes.

    Returns:
        list[dict]: A list of waypoint dictionaries, where each dictionary contains:
                    'pos' (np.ndarray): The [x, y, z] position.
                    'quat' (np.ndarray): The [qx, qy, qz, qw] orientation quaternion.
    """
    waypoints = []
    
    # Create a parameter 't' that goes from 0 to 1
    t = np.linspace(0, 1, num_points)

    # --- Parameterize the Spherical Coordinates ---
    # Phi (polar angle) goes from 0 (north pole) to pi/2 (equator)
    phi = t * np.pi / 2
    
    # Theta (azimuthal angle) makes a number of full revolutions
    theta = t * num_revolutions * 2 * np.pi

    # --- Convert Spherical to Cartesian Coordinates ---
    # This gives the position of each point on the sphere's surface
    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]

    # Assemble the waypoints with workspace validation and cube visibility
    valid_waypoints = 0
    for i in range(num_points):
        eef_pos = np.array([x[i], y[i], z[i]])
        
        # Check if position is within workspace
        if is_in_workspace(eef_pos, center, max_radius=0.6):
            # Check if cube is visible from this position
            if is_cube_visible(eef_pos, cube_pos, camera_fov_deg=75, min_distance=0.10, max_distance=0.8):
                # Calculate direction vector from end-effector to cube
                direction_to_cube = cube_pos - eef_pos
                direction_to_cube = direction_to_cube / np.linalg.norm(direction_to_cube)
                
                # Calculate quaternion to orient camera Z-axis toward cube
                # Default camera Z-axis is [0, 0, -1] (pointing down)
                default_z = np.array([0, 0, -1])
                target_z = direction_to_cube
                
                # Calculate rotation quaternion
                quat = quaternion_from_vectors(default_z, target_z)
                
                waypoints.append({'pos': eef_pos, 'quat': quat})
                valid_waypoints += 1
            else:
                print(f"Skipping waypoint {i}: cube not visible from position {eef_pos}")
        else:
            print(f"Skipping waypoint {i}: position {eef_pos} is outside workspace")
    
    print(f"Generated {valid_waypoints} valid waypoints out of {num_points} requested")
    return waypoints

def quaternion_from_vectors(vec1, vec2):
    """
    Calculate quaternion to rotate from vec1 to vec2.
    
    Args:
        vec1 (np.ndarray): Source vector
        vec2 (np.ndarray): Target vector
    
    Returns:
        np.ndarray: Quaternion [qx, qy, qz, qw]
    """
    # Normalize vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Calculate rotation axis and angle
    axis = np.cross(vec1, vec2)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:  # Vectors are parallel
        if np.dot(vec1, vec2) > 0:  # Same direction
            return np.array([0, 0, 0, 1])  # Identity quaternion
        else:  # Opposite direction
            # Find perpendicular axis
            if abs(vec1[0]) < 0.9:
                axis = np.cross(vec1, [1, 0, 0])
            else:
                axis = np.cross(vec1, [0, 1, 0])
            axis = axis / np.linalg.norm(axis)
            return np.array([axis[0], axis[1], axis[2], 0])  # 180 degree rotation
    
    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
    
    # Convert to quaternion
    half_angle = angle / 2.0
    qw = np.cos(half_angle)
    qx = axis[0] * np.sin(half_angle)
    qy = axis[1] * np.sin(half_angle)
    qz = axis[2] * np.sin(half_angle)
    
    return np.array([qx, qy, qz, qw])

# --- Data Collection Logic ---
print("Starting spherical spiral eye-in-hand data collection for Panda Robot with WipingGripper in Lift Environment...")
obs = env.reset()
frames_data = []

cam_id = env.sim.model.camera_name2id(CAMERA_NAME)
fovy = env.sim.model.cam_fovy[cam_id]
fy = IMAGE_HEIGHT / (2 * np.tan(np.deg2rad(fovy) / 2))
fx = IMAGE_WIDTH / (2 * np.tan(np.deg2rad(fovy) / 2))

# Get cube position from environment
cube_pos = obs['cube_pos']
print(f"Cube position: {cube_pos}")

# Generate spiral waypoints with cube visibility constraint
waypoints = generate_spherical_spiral_waypoints(
    radius=SPHERE_RADIUS,
    center=SPHERE_CENTER,
    cube_pos=cube_pos,
    num_points=NUM_POINTS,
    num_revolutions=NUM_REVOLUTIONS
)

frame_count = 0

# --- Spiral Motion Loop ---
for i, waypoint in enumerate(waypoints):
    target_pos = waypoint['pos']
    target_quat = waypoint['quat']
    
    # Move to target position (orientation control would need more complex implementation)
    eef_pos = obs['robot0_eef_pos']
    pos_error = target_pos - eef_pos
    error_magnitude = np.linalg.norm(pos_error)
    
    # Check if target is too far away (safety check)
    if error_magnitude > 0.5:
        print(f"Warning: Waypoint {i} too far from current position. Error: {error_magnitude:.3f}")
        # Scale down the action to prevent large jumps
        action = np.zeros(env.action_dim)
        action[:3] = pos_error * 2.0  # Reduced gain
    else:
        action = np.zeros(env.action_dim)
        action[:3] = pos_error * 5
    # No gripper action needed for wiping gripper (it's passive)

    # Step the simulation with error handling
    for step in range(10):
         obs, reward, done, info = env.step(action)
         # Check if robot is behaving normally
         current_eef_pos = obs['robot0_eef_pos']
         if np.linalg.norm(current_eef_pos) > 2.0:  # Robot went too far
             print(f"Robot went out of control at waypoint {i}. Resetting...")
             obs = env.reset()
             break
         # No need to check for episode termination since we set ignore_done=True
    
    # Get images from selected cameras only
    eye_in_hand_img = obs[f"{CAMERA_NAME}_image"]
    agentview_img = obs["agentview_image"]
    frontview_img = obs["frontview_image"]
    sideview_img = obs["sideview_image"]
    
    # Convert images to BGR for OpenCV display
    images = [eye_in_hand_img, agentview_img, frontview_img, sideview_img]
    processed_images = [cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR) for img in images]
    
    # Create a 2x2 grid layout (4 cameras)
    top_row = np.hstack([processed_images[0], processed_images[1]])
    bottom_row = np.hstack([processed_images[2], processed_images[3]])
    combined_feed = np.vstack([top_row, bottom_row])
    
    # Add labels to each camera view
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    
    cv2.putText(combined_feed, "Eye-in-Hand", (10, 30), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Agent View", (266, 30), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Front View", (10, 286), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Side View", (266, 286), font, font_scale, color, thickness)
    
    # Save the frame data (only eye-in-hand for dataset)
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

    # Live Feed - Show all camera views
    cv2.imshow("Panda Robot Trajectory - Multi-Camera View", combined_feed)
    cv2.waitKey(1)

    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Collected frame {frame_count}/{NUM_POINTS}")

# --- Add single agentview image and transform at the end ---
print("Adding single agentview image and transform...")

# Get the final observation to capture agentview
final_obs = obs  # Use the last observation from the simulation

# Get agentview camera pose
agentview_cam_id = env.sim.model.camera_name2id("agentview")
agentview_cam_pos = env.sim.data.cam_xpos[agentview_cam_id]
agentview_cam_rot = env.sim.data.cam_xmat[agentview_cam_id].reshape(3, 3)

# Create agentview camera-to-world matrix
agentview_c2w_matrix = np.eye(4)
agentview_c2w_matrix[:3, :3] = agentview_cam_rot
agentview_c2w_matrix[:3, 3] = agentview_cam_pos
correction = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
agentview_c2w_matrix = agentview_c2w_matrix @ correction

# Save single agentview image
agentview_img = final_obs["agentview_image"]
agentview_filename = "agentview_reference.png"
agentview_path = os.path.join(AGENTVIEW_DIR, agentview_filename)
agentview_bgr = cv2.cvtColor(np.flipud(agentview_img), cv2.COLOR_RGB2BGR)
cv2.imwrite(agentview_path, agentview_bgr)

# Add agentview to frames data
frames_data.append({
    "file_path": f"agentview_images/{agentview_filename}", 
    "transform_matrix": agentview_c2w_matrix.tolist()
})

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
print(f"\nSpherical spiral data collection complete! Dataset saved in '{OUTPUT_DIR}'")
print(f"Collected {NUM_POINTS} images from Panda Robot eye-in-hand camera")
print(f"Collected 1 reference image from Panda Robot agentview camera")
print(f"Total: {len(frames_data)} camera poses for 3D Gaussian Splatting")
print(f"Spiral path: {NUM_REVOLUTIONS} revolutions around sphere of radius {SPHERE_RADIUS}m")
print(f"Environment: Lift with red cube and WipingGripper")
print(f"Dataset structure:")
print(f"  - images/: {NUM_POINTS} eye-in-hand camera images (spiral trajectory)")
print(f"  - agentview_images/: 1 reference agentview image (stationary perspective)")
print(f"  - transforms.json: camera poses for all cameras")
