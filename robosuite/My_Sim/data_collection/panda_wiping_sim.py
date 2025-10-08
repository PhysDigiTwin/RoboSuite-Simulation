import robosuite as suite
from robosuite.environments.manipulation.wipe import Wipe
import numpy as np
import cv2
import os

# Custom Wipe environment class with additional cameras
class PandaWipingEnv(Wipe):
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

# Create the environment using the existing Wipe environment
env = PandaWipingEnv(
    robots="Panda",
    gripper_types="WipingGripper",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    control_freq=20,
    camera_names=["robot0_eye_in_hand", "agentview", "frontview", "sideview"],
    camera_heights=256,
    camera_widths=256,
)

# Create output directory for saving images
output_dir = "panda_wiping_images"
os.makedirs(output_dir, exist_ok=True)

print("Panda Robot with Wiping Gripper Simulation")
print("Press 'q' to quit, 's' to save images, 'r' to reset")

# Reset environment
obs = env.reset()

# Simple scripted policy for wiping motion
step_count = 0
max_steps = 1000

while step_count < max_steps:
    # Get current state - Wipe environment has different observation keys
    eef_pos = obs['robot0_eef_pos']
    
    # Simple wiping motion: move in a circular pattern over the table
    # The Wipe environment has a table at z=0.9, so we'll wipe over it
    angle = (step_count * 0.02) % (2 * np.pi)
    radius = 0.15
    center_x, center_y = 0.15, 0.0  # Center of the table
    
    target_x = center_x + radius * np.cos(angle)
    target_y = center_y + radius * np.sin(angle)
    target_z = 0.95  # Just above the table surface
    
    target_pos = np.array([target_x, target_y, target_z])
    
    # Compute action to move towards target
    pos_error = target_pos - eef_pos
    action = np.zeros(env.action_dim)
    action[:3] = pos_error * 2.0  # Position control
    
    # Step the simulation
    obs, reward, done, info = env.step(action)
    
    # Get images from all cameras
    eye_in_hand_img = obs["robot0_eye_in_hand_image"]
    agentview_img = obs["agentview_image"]
    frontview_img = obs["frontview_image"]
    sideview_img = obs["sideview_image"]
    
    # Convert images to BGR for OpenCV display
    images = [eye_in_hand_img, agentview_img, frontview_img, sideview_img]
    processed_images = [cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR) for img in images]
    
    # Create a 2x2 grid layout
    top_row = np.hstack([processed_images[0], processed_images[1]])
    bottom_row = np.hstack([processed_images[2], processed_images[3]])
    combined_feed = np.vstack([top_row, bottom_row])
    
    # Add labels to each camera view
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2
    
    cv2.putText(combined_feed, "Eye-in-Hand", (10, 30), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Agent View", (266, 30), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Front View", (10, 286), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Side View", (266, 286), font, font_scale, color, thickness)
    
    # Display the combined feed
    cv2.imshow("Panda Robot with Wiping Gripper", combined_feed)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save individual camera images
        cv2.imwrite(os.path.join(output_dir, f"robot0_eye_in_hand_{step_count:04d}.png"), processed_images[0])
        cv2.imwrite(os.path.join(output_dir, f"agentview_{step_count:04d}.png"), processed_images[1])
        cv2.imwrite(os.path.join(output_dir, f"frontview_{step_count:04d}.png"), processed_images[2])
        cv2.imwrite(os.path.join(output_dir, f"sideview_{step_count:04d}.png"), processed_images[3])
        # Also save the combined view
        cv2.imwrite(os.path.join(output_dir, f"combined_view_{step_count:04d}.png"), combined_feed)
        print(f"Images saved to '{output_dir}' folder!")
    elif key == ord('r'):
        obs = env.reset()
        step_count = 0
        print("Environment reset!")
    
    step_count += 1
    
    if done:
        obs = env.reset()
        step_count = 0
        print("Episode finished. Resetting environment.")

# Cleanup
cv2.destroyAllWindows()
env.close()
print("Simulation complete!")
