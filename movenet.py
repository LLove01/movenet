import tensorflow as tf
import tensorflow_hub as hub
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

from helpers import *

# Load MoveNet (Lightning or Thunder)
model_name = "movenet_lightning"  # or "movenet_thunder"
model = hub.load(
    "https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-lightning/versions/4")
input_size = 192 if model_name == "movenet_lightning" else 256

# Tkinter window setup
window = tk.Tk()
window.title("MoveNet Pose Detection")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Tkinter Canvas
canvas = tk.Canvas(window, width=cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
canvas.pack()


def draw_connections(frame, keypoints, edges, edge_colors, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in zip(edges, edge_colors):
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if c1 > confidence_threshold and c2 > confidence_threshold:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


def draw_keypoints(frame, keypoints, confidence_threshold=0.4):
    y, x, _ = frame.shape
    # Loop through all keypoints to draw them on the frame
    for kp in keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx * x), int(ky * y)), 6,
                       (255, 0, 255), -1)  # Magenta for keypoints


def run_movenet(input_image):
    """Runs MoveNet on a single image and returns keypoints."""
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.image.resize_with_pad(
        input_image, input_size, input_size)

    # Convert image to int32, as expected by the loaded model
    input_image = tf.cast(input_image, dtype=tf.int32)

    # Run model inference.
    outputs = model.signatures['serving_default'](input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    # We take [0, 0] since the batch size and number of detections per image are both 1
    return keypoints_with_scores[0, 0]


def update_canvas():
    global tk_image
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = tf.convert_to_tensor(frame_rgb)
        input_image = tf.expand_dims(input_image, axis=0)

        keypoints_with_scores = run_movenet(input_image)

        # Separate the keypoints and scores
        keypoints = keypoints_with_scores[:, :3]
        draw_keypoints(frame, keypoints)

        # Use a defined confidence threshold for drawing connections
        confidence_threshold = 0.4
        # Inside your update_canvas function
        draw_connections(frame, keypoints, LIMBS,
                         LIMB_COLORS, confidence_threshold)

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tk_image = ImageTk.PhotoImage(image=pil_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image  # Keep a reference to the image object

    window.after(10, update_canvas)


# Global variable for the Tkinter image
tk_image = None

# Start the update process and Tkinter main loop
update_canvas()
window.mainloop()

# Release resources
cap.release()
