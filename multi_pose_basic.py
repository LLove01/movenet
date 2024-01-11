import tensorflow as tf
import tensorflow_hub as hub
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

# Load the multi-pose MoveNet model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

color_map = {
    'm': (255, 0, 255),  # Magenta
    'c':  (255, 255, 0),  # Cyan
    'y':  (0, 255, 255),  # Yellow
}


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color_code in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            # Default to red if color_code not found
            color = color_map.get(color_code, (0, 0, 255))
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


def run_movenet(input_image):
    """Runs MoveNet on a single image and returns keypoints for multiple persons."""
    # Resize and pad the image
    input_image = tf.image.resize_with_pad(input_image, 192, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)

    # Run model inference
    outputs = movenet(input_image)
    keypoints_with_scores = outputs['output_0'].numpy()[
        :, :, :51].reshape((6, 17, 3))
    return keypoints_with_scores


# Tkinter window setup
window = tk.Tk()
window.title("MoveNet MultiPose Detection")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Tkinter Canvas
canvas = tk.Canvas(window, width=cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
canvas.pack()


def update_canvas():
    global tk_image
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = tf.convert_to_tensor(frame_rgb)
        input_image = tf.expand_dims(input_image, axis=0)

        keypoints_with_scores = run_movenet(input_image)

        # Loop through each person detected and render
        for person in keypoints_with_scores:
            draw_connections(frame, person, EDGES, 0.4)
            draw_keypoints(frame, person, 0.4)

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
