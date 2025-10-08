import robosuite as suite
from robosuite.environments.manipulation.lift import Lift
import numpy as np
import cv2
import os

# Custom environment class with multiple cameras
class LiftMultiCam(Lift):
    def _setup_cameras(self):
        super()._setup_cameras()

        # Define the properties for our overhead cameras - much closer for zoomed view
        radius = 0.3  # Much closer to the scene
        height = 0.8  # Lower angle for more detail
        
        # Base quaternion for a camera looking down and inward from the front
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

# Create environment with multiple cameras
env = LiftMultiCam(
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    control_freq=20,
    camera_names=["robot0_eye_in_hand", "agentview", "frontview", "sideview", "birdview"],
    camera_heights=256,
    camera_widths=256,
)

# Reset environment to get initial observation
obs = env.reset()

# Create output directory for saving images
output_dir = "camera_images"
os.makedirs(output_dir, exist_ok=True)

print("Displaying static images from four camera views...")
print("Press 'q' to quit")
print("Press 's' to save current images")

while True:
    # Get images from all cameras
    eye_in_hand_img = obs["robot0_eye_in_hand_image"]
    agentview_img = obs["agentview_image"]
    frontview_img = obs["frontview_image"]
    sideview_img = obs["sideview_image"]
    birdview_img = obs["birdview_image"]
    
    # Convert images to BGR for OpenCV display
    images = [eye_in_hand_img, agentview_img, frontview_img, sideview_img]
    processed_images = []
    
    # Convert images to BGR for OpenCV display
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
    
    # Add text labels
    cv2.putText(combined_feed, "Eye-in-Hand", (10, 30), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Agent View", (266, 30), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Front View", (10, 286), font, font_scale, color, thickness)
    cv2.putText(combined_feed, "Side View", (266, 286), font, font_scale, color, thickness)

    cv2.imwrite(os.path.join(output_dir, "robot0_eye_in_hand.png"), processed_images[0])
    cv2.imwrite(os.path.join(output_dir, "agentview.png"), processed_images[1])
    cv2.imwrite(os.path.join(output_dir, "frontview.png"), processed_images[2])
    cv2.imwrite(os.path.join(output_dir, "sideview.png"), processed_images[3])
    # Also save the combined view
    cv2.imwrite(os.path.join(output_dir, "combined_view.png"), combined_feed)
    print(f"Images saved to '{output_dir}' folder!")

    # Display the combined feed
    cv2.imshow("Static Camera Views", combined_feed)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
env.close()
print("Exiting...")