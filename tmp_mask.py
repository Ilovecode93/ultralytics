import cv2
import torch
from ultralytics import YOLO
import numpy as np
# Load the YOLOv8 model
model = YOLO("/home/deepl/ultralytics_626/712_新loss/weights/best.pt")

# Load the image
image = cv2.imread("/media/deepl/8206e4e5-7eb6-48c5-969a-e61fbc5adff7/Bigpig/9_22_pig/9_22_pig_58.png")

# Run the model
results = model(image)
image_path = "/media/deepl/8206e4e5-7eb6-48c5-969a-e61fbc5adff7/Bigpig/9_22_pig/9_22_pig_58.png"
# Load the image
img = cv2.imread(image_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB



img_height, img_width = img.shape[:2]

# Define a color map for different classes (extend as needed)
color_palette = [
    [255, 0, 0],    # Red
    [0, 0, 255],    # Blue
    [255, 255, 0],  # Cyan
    [0, 255, 255],  # Yellow
    [128, 0, 0],    # Maroon
    [0, 128, 0],    # Dark Green
    [0, 0, 128],    # Navy
    # Add more colors as needed
]
instance_counter = 0
# Iterate through each result (each image)
for result in results:
    # Check if masks are available
    if result.masks is not None:
        masks = result.masks  # Masks object
        boxes = result.boxes  # Boxes object

        # Convert masks to binary masks
        binary_masks = masks.data.cpu().numpy()  # Shape: (num_instances, mask_height, mask_width)

        # Iterate through each detected instance
        for idx in range(len(binary_masks)):
            mask = binary_masks[idx]

            # Resize the mask to match the original image size
            mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

            # Binarize the resized mask
            binary_mask = mask_resized > 0.5  # Threshold can be adjusted

            # Extract class and confidence for the current instance
            box = boxes[idx]
            cls = int(box.cls.cpu().numpy())
            conf = box.conf.cpu().numpy()

            # Get class name
            class_name = model.names[cls] if cls < len(model.names) else f'Class {cls}'

            # Get color for the class
            # class_color = color_map.get(cls, [255, 255, 255])  # Default to white if class not in color_map
            color = color_palette[instance_counter % len(color_palette)]
            # Create a colored mask based on class
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            assigned_color = color
            colored_mask[binary_mask] = assigned_color
            instance_counter += 1  # Increment the counter

            # Blend the mask with the image
            img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten().astype(int)


            mask_center_y, mask_center_x = np.where(binary_mask)
            mask_center_x = int(np.mean(mask_center_x))
            mask_center_y = int(np.mean(mask_center_y))
            # Create label
            # label = f'{class_name} {conf:.2f}'
            label = "pig"
            count_pig = idx + 1
            # Draw bounding box
            # cv2.rectangle(img, (x1, y1), (x2, y2), class_color, 2)

            # Put label above the bounding box
            cv2.putText(img, str(count_pig), (mask_center_x, mask_center_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), 2)
    else:
        print("No masks detected in the image.")

# Convert back to BGR for saving with OpenCV
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite('output_with_masks.jpg', img_bgr)