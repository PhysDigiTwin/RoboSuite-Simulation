import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
import numpy as np
import cv2
import os
import json
import shutil

# --- Setup ---
OUTPUT_DIR = "robosuite_baxter_lift_spiral_dataset"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Spiral Parameters ---
SPHERE_RADIUS = 0.6  # Larger radius for Baxter's bigger workspace
SPHERE_CENTER = np.array([0.0, 0.0, 0.9])  # Center around the cube
NUM_POINTS = 360  # Total number of waypoints
NUM_REVOLUTIONS = 3  # Number of spiral revolutions

# --- Environment Definition ---
CAMERA_NAME = "robot0_eye_in_hand"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# Custom Lift environment class with additional cameras
class JacoLiftOrbitEnv(Lift):
    def _setup_cameras(self):
        """Set up cameras for visualization"""
        super()._setup_cameras()
        
        # Add custom cameras for better viewing
        self.sim.model.camera_add(
            "frontview",
            pos=[0.8, 0, 1.4],
            quat=[0.383, 0, 0.924, 0],
        )
        
        self.sim.model.camera_add(
            "sideview", 
            pos=[-0.4, 0.7, 1.4],
            quat=[0.191, -0.800, 0.462, 0.332],
        )
        
        self.sim.model.camera_add(
            "birdview",
            pos=[0.0, 0.0, 2.0],  # High above looking down
            quat=[0.707, 0, 0, 0.707],  # Looking straight down
        )

env = JacoLiftOrbitEnv(
    robots="Jaco",
    gripper_types="WipingGripper",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    control_freq=20,
    camera_names=[CAMERA_NAME, "agentview", "frontview", "sideview", "birdview"],
    camera_heights=IMAGE_HEIGHT,
    camera_widths=IMAGE_WIDTH,
    horizon=10000,  # Very large horizon for data collection
    ignore_done=True,  # Ignore episode termination
)

def is_in_workspace(pos, center, max_radius=0.7):
    """
    Check if a position is within the robot's workspace.
    Jaco has a good workspace - similar to Sawyer but more reliable.
    
    Args:
        pos (np.ndarray): Position to check
        center (np.ndarray): Workspace center
        max_radius (float): Maximum reachable radius
    
    Returns:
        bool: True if position is reachable
    """
    distance = np.linalg.norm(pos - center)
    return distance <= max_radius

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

def generate_spherical_spiral_waypoints(radius=0.6, center=np.array([0, 0, 0]), cube_pos=np.array([0, 0, 0.9]), num_points=200, num_revolutions=5):
    """
    Generates waypoints for a spiral path on the upper hemisphere of a sphere.

    The orientation is calculated so the end-effector's Z-axis points towards
    the cube position, ensuring the cube is always in view.

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
        
        # Check if position is within workspace (Jaco has good workspace)
        if is_in_workspace(eef_pos, center, max_radius=0.7):
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
            print(f"Skipping waypoint {i}: position {eef_pos} is outside workspace")
    
    print(f"Generated {valid_waypoints} valid waypoints out of {num_points} requested")
    return waypoints

# --- Data Collection Logic ---
print("Starting spherical spiral eye-in-hand data collection for Jaco Robot with WipingGripper in Lift Environment...")
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
    if error_magnitude > 0.5:  # Jaco can handle moderate movements
        print(f"Warning: Waypoint {i} too far from current position. Error: {error_magnitude:.3f}")
        # Scale down the action to prevent large jumps
        action = np.zeros(env.action_dim)
        action[:3] = pos_error * 3.0  # Reduced gain
    else:
        action = np.zeros(env.action_dim)
        action[:3] = pos_error * 5
    # No gripper action needed for wiping gripper (it's passive)

    # Step the simulation with error handling
    for step in range(10):
         obs, reward, done, info = env.step(action)
         # Check if robot is behaving normally
         current_eef_pos = obs['robot0_eef_pos']
         if np.linalg.norm(current_eef_pos) > 2.0:  # Jaco has good reach
             print(f"Robot went out of control at waypoint {i}. Resetting...")
             obs = env.reset()
             break
         # No need to check for episode termination since we set ignore_done=True
    
    # Get images from all cameras
    eye_in_hand_img = obs[f"{CAMERA_NAME}_image"]
    agentview_img = obs["agentview_image"]
    frontview_img = obs["frontview_image"]
    sideview_img = obs["sideview_image"]
    birdview_img = obs["birdview_image"]
    
    # Convert images to BGR for OpenCV display
    images = [eye_in_hand_img, agentview_img, frontview_img, sideview_img, birdview_img]
    processed_images = [cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR) for img in images]
    
    # Create a 2x3 grid layout (5 cameras + 1 empty space)
    top_row = np.hstack([processed_images[0], processed_images[1], processed_images[2]])
    bottom_row = np.hstack([processed_images[3], processed_images[4], np.zeros_like(processed_images[0])])
    combined_feed = np.vstack([top_row, bottom_row])
    
    # Add labels to each camera view
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    
    cv2.putText(combined_feed, "Eye-in-Hand", (10, 30), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Agent View", (266, 30), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Front View", (522, 30), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Side View", (10, 286), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Bird View", (266, 286), font, font_scale, color, thickness)
    
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
    cv2.imshow("Jaco Robot Trajectory - Multi-Camera View", combined_feed)
    cv2.waitKey(1)

    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Collected frame {frame_count}/{NUM_POINTS}")

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
print(f"Collected {NUM_POINTS} images from Jaco Robot eye-in-hand camera")
print(f"Spiral path: {NUM_REVOLUTIONS} revolutions around sphere of radius {SPHERE_RADIUS}m")
print(f"Environment: Lift with red cube and WipingGripper")
print(f"Robot: Jaco (reliable dexterity and workspace)")
