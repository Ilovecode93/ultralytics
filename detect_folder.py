import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Batch process images with YOLOv8 and apply mask overlays.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input images folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to save the output images.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLOv8 model weights (e.g., best.pt).')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detections (default: 0.25).')
    parser.add_argument('--extensions', type=str, nargs='+', default=['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'],
                        help='List of image file extensions to process (default: .png .jpg .jpeg .bmp .tif .tiff).')
    args = parser.parse_args()
    return args

def load_model(model_path, device):
    try:
        model = YOLO(model_path).to(device)
        print(f"Model loaded successfully from {model_path}")
        print('Class Names:', model.names)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        exit(1)

def get_image_files(input_folder, extensions):
    image_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(tuple(extensions)):
                image_files.append(os.path.join(root, file))
    return image_files

def create_color_palette():
    """Define a color palette for different instances."""
    return [
        [255, 0, 0],      # Red
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Cyan
        [0, 255, 255],    # Yellow
        [128, 0, 0],      # Maroon
        [0, 128, 0],      # Dark Green
        [0, 0, 128],      # Navy
        [255, 165, 0],    # Orange
        [128, 128, 0],    # Olive
        [128, 0, 128],    # Purple
        # Add more colors as needed
    ]

def process_image(image_path, model, output_folder, color_palette, conf_threshold, input_folder):

    try:
        # Load the image
        print("------image_path ---------", image_path)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to read image {image_path}. Skipping.")
            return

        original_img = img.copy()

        img_height, img_width = img.shape[:2]

        # Run the model with specified confidence threshold
        results = model(original_img, conf=conf_threshold)

        instance_counter = 0  # Counter for color assignment
        num_instances = 0
        # Iterate through each result (YOLO can process multiple images, but here it's one)
        for result in results:
            # Check if masks are available
            if result.masks is not None and len(result.masks.data) > 0:
                masks = result.masks.data.cpu().numpy()  # Shape: (num_instances, mask_height, mask_width)
                boxes = result.boxes  # Boxes object

                num_instances = masks.shape[0]

                for idx in range(num_instances):
                    mask = masks[idx]

                    # Resize the mask to match the original image size
                    mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

                    # Binarize the resized mask
                    binary_mask = mask_resized > 0.5  # Threshold can be adjusted

                    # Extract class and confidence for the current instance
                    box = boxes[idx]
                    cls = int(box.cls.cpu().numpy())
                    conf = box.conf.cpu().numpy()

                    # Get class name

                    # Get color for the instance
                    color = color_palette[instance_counter % len(color_palette)]
                    instance_counter += 1  # Increment the counter

                    # Create a colored mask based on class
                    colored_mask = np.zeros_like(img, dtype=np.uint8)
                    colored_mask[binary_mask] = color

                    # Blend the mask with the image
                    img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten().astype(int)

                    # Calculate mask center
                    mask_center_y, mask_center_x = np.where(binary_mask)
                    mask_center_x = int(np.mean(mask_center_x))
                    mask_center_y = int(np.mean(mask_center_y))

                    # Labeling
                    count_pig = idx + 1
                    # Alternatively, use a custom label:
                    # label = "pig"

                    # Put label above the bounding box
                    cv2.putText(img, str(count_pig), (mask_center_x, mask_center_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 0), 2, cv2.LINE_AA)

            else:
                print(f"No masks detected in the image {image_path}.")

        # Prepare output path
        # relative_path = os.path.relpath(image_path, start=os.path.commonpath([image_path, input_folder]))
        # output_image_path = os.path.join(output_folder, relative_path)
        # output_dir = os.path.dirname(output_image_path)
        # os.makedirs(output_dir, exist_ok=True)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 5
        font_color = (0, 0, 255)  # 红色 (BGR 格式)
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(str(num_instances), font, font_scale, font_thickness)

        # 计算文本的位置（在图片的正上方）
        text_x = int((img_width - text_width) / 2)
        text_y = int(text_height + 10)  # 稍微偏下一点，避免文字太靠近边缘

        # 在图片上写数字
        cv2.putText(img, str(num_instances), (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        combined_img = np.hstack((original_img, img))
        # Save the annotated image
        # cv2.imwrite(output_image_path, combined_img)
        
        if num_instances > 0:
            # for idx in range(num_instances):
            count_pig = num_instances
            count_pig_folder = os.path.join(output_folder, f"count_pig_{count_pig}")
            os.makedirs(count_pig_folder, exist_ok=True)
            count_pig_image_path = os.path.join(count_pig_folder, os.path.basename(image_path))
            cv2.imwrite(count_pig_image_path, combined_img)
        else:
            count_pig = 0
            count_pig_folder = os.path.join(output_folder, f"count_pig_{count_pig}")
            os.makedirs(count_pig_folder, exist_ok=True)
            count_pig_image_path = os.path.join(count_pig_folder, os.path.basename(image_path))
            cv2.imwrite(count_pig_image_path, combined_img)
            
        
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    args = parse_arguments()

    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder {args.input_folder} does not exist or is not a directory.")
        exit(1)

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load YOLO model
    model = load_model(args.model_path, device)

    # Get list of image files
    image_files = get_image_files(args.input_folder, args.extensions)
    if not image_files:
        print(f"No images found in {args.input_folder} with extensions {args.extensions}.")
        exit(0)

    print(f"Found {len(image_files)} images to process.")

    # Create color palette
    color_palette = create_color_palette()

    # Process each image with a progress bar
    for image_path in tqdm(image_files, desc="Processing Images"):
        process_image(image_path, model, args.output_folder, color_palette, args.conf_threshold, args.input_folder)

    print("Processing completed.")

if __name__ == "__main__":
    main()
