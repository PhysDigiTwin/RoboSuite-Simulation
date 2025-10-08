import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
import numpy as np
import cv2

# 1. Define the custom environment with the new 4-camera setup
class LiftMultiCam(Lift):
    """
    A Lift environment with one agentview and three overhead cameras
    arranged equidistantly.
    """
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


# 2. Instantiate the class, making sure to request all four camera names
env = LiftMultiCam(
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["robot0_eye_in_hand", "frontview", "sideview", "birdview"],
    camera_heights=256,
    camera_widths=256,
)

# Reset the environment
obs = env.reset()

# --- Simulation Loop ---
for i in range(1000):
    
    low, high = env.action_spec
    action = np.random.uniform(low=low, high=high)
    obs, reward, done, info = env.step(action)

    # 3. Retrieve all four camera images
    agentview_img = obs["robot0_eye_in_hand_image"]
    cam_top_front_img = obs["frontview_image"]
    cam_top_left_img = obs["sideview_image"]
    cam_top_right_img = obs["birdview_image"]

    # Process all four images for display
    images = [agentview_img, cam_top_front_img, cam_top_left_img, cam_top_right_img]
    processed_images = [cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR) for img in images]
    
    # 4. Arrange the four images into a 2x2 grid for display
    top_row = np.hstack([processed_images[0], processed_images[1]])
    bottom_row = np.hstack([processed_images[2], processed_images[3]])
    combined_feed = np.vstack([top_row, bottom_row])
    
    cv2.imshow("Robosuite 4-Camera Grid", combined_feed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if done:
        obs = env.reset()

# Cleanup
cv2.destroyAllWindows()
env.close()

print("Simulation finished.")