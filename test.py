import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ==========================================
# STEP 1: Load MoveNet model
# ==========================================
model_path = "3.tflite"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================================
# STEP 2: Load and preprocess image
# ==========================================
img_path = "person.jpg"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at {img_path}")

image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the model's expected input size and type
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]
input_dtype = input_details[0]['dtype']

# Resize the image to the model's expected size
input_img = cv2.resize(image_rgb, (input_width, input_height))

# Add a batch dimension and ensure the data type is correct
# DO NOT normalize by dividing by 255.0
input_img = np.expand_dims(input_img, axis=0).astype(input_dtype)

# ==========================================
# STEP 3: Run inference
# ==========================================
interpreter.set_tensor(input_details[0]['index'], input_img)
interpreter.invoke()

keypoints = interpreter.get_tensor(output_details[0]['index'])[0, 0, :, :]

# ==========================================
# STEP 4: Create pseudo-heatmap function
# ==========================================
h, w, _ = image_rgb.shape

def generate_heatmap(cx, cy, sigma=15):
    """Generate a Gaussian heatmap for a single keypoint"""
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)[:, np.newaxis]
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

# COCO keypoint order
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# ==========================================
# STEP 5: Plot 17 individual heatmaps
# ==========================================
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for i, (y, x, score) in enumerate(keypoints):
    ax = axes[i // 6, i % 6]

    cx, cy = int(x * w), int(y * h)

    # Always generate a heatmap, even if score < 0.3
    heatmap = generate_heatmap(cx, cy, sigma=15) 

    # Overlay on image
    ax.imshow(image_rgb, alpha=0.5)
    ax.imshow(heatmap, cmap="hot", alpha=0.7)

    ax.set_title(f"{keypoint_names[i]} ({score:.2f})")
    ax.axis("off")

plt.tight_layout()
plt.show()
