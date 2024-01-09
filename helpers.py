LIMBS = [
    (5, 6),  # Left shoulder to right shoulder
    (5, 7),  # Left shoulder to left elbow
    (7, 9),  # Left elbow to left wrist
    (6, 8),  # Right shoulder to right elbow
    (8, 10),  # Right elbow to right wrist
    (5, 11),  # Left shoulder to left hip
    (6, 12),  # Right shoulder to right hip
    (11, 12),  # Left hip to right hip
    (11, 13),  # Left hip to left knee
    (13, 15),  # Left knee to left ankle
    (12, 14),  # Right hip to right knee
    (14, 16),  # Right knee to right ankle
    (5, 0),  # Left shoulder to nose
    (6, 0),  # Right shoulder to nose
    (0, 1),  # Nose to left eye
    (0, 2),  # Nose to right eye
    (1, 3),  # Left eye to left ear
    (2, 4),  # Right eye to right ear
]

# Yellow (0, 255, 255)
# cyan (255, 255, 0)

# Define the colors for each limb
LIMB_COLORS = [
    (0, 255, 255),  # Yellow for central line: left shoulder to right shoulder
    (255, 0, 255),  # Magenta for left arm: left shoulder to left elbow
    (255, 0, 255),  # Magenta for left arm: left elbow to left wrist
    (255, 255, 0),  # Cyan for right arm: right shoulder to right elbow
    (255, 255, 0),  # Cyan for right arm: right elbow to right wrist
    (255, 0, 255),  # Magenta for left torso: left shoulder to left hip
    (255, 255, 0),  # Cyan for right torso: right shoulder to right hip
    (0, 255, 255),  # Yellow for central line: left hip to right hip
    (255, 0, 255),  # Magenta for left leg: left hip to left knee
    (255, 0, 255),  # Magenta for left leg: left knee to left ankle
    (255, 255, 0),  # Cyan for right leg: right hip to right knee
    (255, 255, 0),  # Cyan for right leg: right knee to right ankle
    (255, 0, 255),  # Magenta for left shoulder to nose
    (255, 255, 0),  # Cyan for right shoulder to nose
    (255, 0, 255),  # Magenta for nose to left eye
    (255, 255, 0),  # Cyan for nose to right eye
    (255, 0, 255),  # Magenta for left eye to left ear
    (255, 255, 0),  # Cyan for right eye to right ear
]
