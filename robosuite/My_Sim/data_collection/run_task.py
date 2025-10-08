import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
import numpy as np
import cv2
import time

# Using your custom environment class
class LiftMultiCam(Lift):
    def _setup_cameras(self):
        super()._setup_cameras()

        # Define the properties for our overhead cameras
        # A radius of 1.0m from the center, at a height of 1.5m
        radius = 1.0
        height = 1.5
        
        # Base quaternion for a camera looking down and inward from the front
        # This corresponds to a 135-degree rotation around the Y-axis
        base_quat = [0.383, 0, 0.924, 0]

        # Camera 1: Directly in front of the robot, looking back and down
        self.sim.model.camera_add(
            "frontview",
            pos=[radius, 0, height],
            quat=base_quat,
        )

        # Camera 2: To the back-left of the robot, looking forward-right and down
        self.sim.model.camera_add(
            "sideview",
            pos=[-0.5 * radius, 0.866 * radius, height], # 120 degrees around
            quat=[0.191, -0.800, 0.462, 0.332], # Rotated quaternion
        )

        # Camera 3: To the back-right of the robot, looking forward-left and down
        self.sim.model.camera_add(
            "birdview",
            pos=[-0.5 * radius, -0.866 * radius, height], # 240 degrees around
            quat=[0.191, 0.800, 0.462, -0.332], # Rotated quaternion
        )

# Instantiate your custom class
env = LiftMultiCam(
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    control_freq=20, # Use a consistent control frequency
    camera_names=["agentview", "frontview", "sideview", "birdview"],
    camera_heights=256,
    camera_widths=256,
)

obs = env.reset()
# Define a drop zone away from the starting cube position
drop_zone = np.array([0.2, 0.2, 0.9]) 
task_phase = "reaching" # Our state machine starts here

# --- Simulation Loop with Scripted Policy ---
for i in range(2000): # Increased steps for the task
    
    # Get current state from observations
    eef_pos = obs['robot0_eef_pos']
    cube_pos = obs['cube_pos']
    gripper_qpos = obs['robot0_gripper_qpos']
    
    # Initialize action array
    action = np.zeros(env.action_dim)
    action[6] = -5.0
    # --- State Machine Logic for Pick and Place ---
    if task_phase == "reaching":
        # 1. Move gripper above the cube
        target_pos = cube_pos + np.array([0, 0, 0.15]) # Target 15cm above cube
        pos_error = target_pos - eef_pos
        action[:3] = pos_error * 4 # Proportional control to move the arm
        
        # If we are close enough, move to the next phase
        if np.linalg.norm(pos_error) < 0.02:
            task_phase = "grasping"
            
    elif task_phase == "grasping":
        # 2. Move gripper down to the cube
        pos_error = cube_pos - eef_pos
        action[:3] = pos_error * 4
        # print(np.linalg.norm(pos_error))
        # If we are very close, close the gripper
        if np.linalg.norm(pos_error) < 0.005:
            action[6] = 1 # Close gripper

            # Check if gripper is closed (velocity is near zero)
            print(abs(gripper_qpos[0] - gripper_qpos[1]))
            if abs(gripper_qpos[0] - gripper_qpos[1]) > 0.075: # Simple check if it has grasped something
                task_phase = "lifting"

    elif task_phase == "lifting":
        # 3. Lift the cube straight up
        target_pos = eef_pos + np.array([0, 0, 0.2]) # Lift 20cm
        pos_error = target_pos - eef_pos
        action[:3] = pos_error * 3
        action[6] = 1 # Keep gripper closed
        
        # Once lifted high enough, move to drop zone
        if eef_pos[2] > 1.0: # Check if Z-height is above 1.0m
            task_phase = "moving_to_drop"

    elif task_phase == "moving_to_drop":
        # 4. Move to the drop zone
        pos_error = drop_zone - eef_pos
        action[:3] = pos_error * 3
        action[6] = 1 # Keep gripper closed
        
        if np.linalg.norm(pos_error) < 0.02:
            task_phase = "dropping"

    elif task_phase == "dropping":
        # 5. Open the gripper
        action[6] = -1 # Open gripper
        
        # Wait a moment for the block to drop, then reset the task
        time.sleep(0.5)
        task_phase = "reaching" # Restart the cycle
        
    # Step the simulation
    obs, reward, done, info = env.step(action)
    
    # --- Visualization (same as before) ---
    # 3. Retrieve all four camera images
    agentview_img = obs["agentview_image"]
    cam_top_front_img = obs["frontview_image"]
    cam_top_left_img = obs["sideview_image"]
    cam_top_right_img = obs["birdview_image"]
    images = [agentview_img, cam_top_front_img, cam_top_left_img, cam_top_right_img]
    processed_images = [cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR) for img in images]
    top_row = np.hstack([processed_images[0], processed_images[1]])
    bottom_row = np.hstack([processed_images[2], processed_images[3]])
    combined_feed = np.vstack([top_row, bottom_row])
    cv2.imshow("Robosuite Scripted Policy", combined_feed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if done:
        obs = env.reset()
        task_phase = "reaching" # Reset state machine on episode end

# Cleanup
cv2.destroyAllWindows()
env.close()