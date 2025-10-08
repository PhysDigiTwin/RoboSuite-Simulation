import cv2
import os
import glob
import natsort # You may need to install this: pip install natsort

# --- Configuration ---
# Directory where your images were saved
IMAGE_DIR = "robosuite_lift_spiral_dataset/images/" 
# Name of the output video file
OUTPUT_VIDEO_FILE = "lift_spiral_video2.mp4"
# Frames per second for the output video
FRAME_RATE = 30

# --- Main Script ---
print(f"Looking for images in: {IMAGE_DIR}")

# Get all the image file paths
image_files = glob.glob(os.path.join(IMAGE_DIR, '*.png'))

# Sort the files numerically (e.g., frame_1.png, frame_2.png, ..., frame_10.png)
# natsort handles this correctly, whereas a simple sort might do 1, 10, 2.
if not image_files:
    print("Error: No images found in the specified directory.")
    exit()

image_files = natsort.natsorted(image_files)

# Read the first image to get the frame size (width, height)
first_frame = cv2.imread(image_files[0])
height, width, layers = first_frame.shape
frame_size = (width, height)

# Initialize the VideoWriter object
# The 'mp4v' codec is a good choice for creating .mp4 files.
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, FRAME_RATE, frame_size)

print(f"Found {len(image_files)} images. Creating video...")

# Loop through all the image files and write them to the video
for filename in image_files:
    img = cv2.imread(filename)
    out.write(img)

# Release the VideoWriter and clean up
out.release()

print(f"\nVideo creation complete! File saved as: {OUTPUT_VIDEO_FILE}")