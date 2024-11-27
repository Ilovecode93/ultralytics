import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import argparse
from tqdm import tqdm
import csv
import re  

def parse_arguments():
    parser = argparse.ArgumentParser(description='Batch process images with YOLOv8 and apply mask overlays.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input images folder.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLOv8 model weights (e.g., best.pt).')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detections (default: 0.25).')
    parser.add_argument('--extensions', type=str, nargs='+', default=['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'],
                        help='List of image file extensions to process (default: .png .jpg .jpeg .bmp .tif .tiff).')
    parser.add_argument('--output_csv', type=str, default='results.csv', help='Path to the output CSV file (default: results.csv).')  # Optional: Allow specifying CSV path
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
            if file.lower().endswith(tuple(extensions)) and file != "原始图片.jpg":
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

def process_image(image_path, model, color_palette, conf_threshold, input_folder):
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to read image {image_path}. Skipping.")
            return

        pts = np.array([(712.8, 782.4), (710.4, 1209.6), (1168.8, 1195.2), (1135.2, 744.0), (712.8, 780.0)], np.int32)
        original_img = img.copy()
        mask = np.zeros_like(img[:, :, 0])
        cv2.fillPoly(mask, [pts], 255)
        masked_img = cv2.bitwise_and(img, img, mask=mask)        
        

        img_height, img_width = img.shape[:2]
        # Run the model with specified confidence threshold
        results = model(masked_img, conf=conf_threshold)

        instance_counter = 0
        num_instances = 0
        mask_pixel_counts = []  # 存储每个mask包含的像素数量
        count_pig = 0
        for result in results:
            if result.masks is not None and len(result.masks.data) > 0:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes

                num_instances = masks.shape[0]

                for idx in range(num_instances):
                    mask = masks[idx]
                    # 重置mask大小
                    mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                    # 二值化mask
                    binary_mask = mask_resized > 0.5

                    # 计算原始图像中被mask覆盖区域的像素值总和
                    pixel_count = np.sum(binary_mask)  # 计算二值mask中为True的像素数量
                    mask_pixel_counts.append({
                        'instance_id': idx + 1,
                        'pixel_count': int(pixel_count)  # 转换为整数
                    })

                    # 获取颜色
                    color = color_palette[instance_counter % len(color_palette)]
                    instance_counter += 1

                    # 创建彩色mask
                    colored_mask = np.zeros_like(img, dtype=np.uint8)
                    colored_mask[binary_mask] = color

                    # 混合mask和图像
                    img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

                    box = boxes[idx]
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten().astype(int)

                    # 计算mask中心
                    mask_center_y, mask_center_x = np.where(binary_mask)
                    mask_center_x = int(np.mean(mask_center_x))
                    mask_center_y = int(np.mean(mask_center_y))

                    # 添加标签
                    count_pig = idx + 1
                    cv2.putText(img, str(count_pig), (mask_center_x, mask_center_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)


            else:
                print(f"No masks detected in the image {image_path}.")

        # 添加总数文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 5
        font_color = (0, 0, 255)
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(str(num_instances), font, font_scale, font_thickness)

        text_x = int((img_width - text_width) / 2)
        text_y = int(text_height + 10)

        cv2.putText(img, str(num_instances), (text_x, text_y), font, 
                    font_scale, font_color, font_thickness, cv2.LINE_AA)
        
        combined_img = np.hstack((original_img, img))

        tmp_folder = image_path.rsplit('/', 1)[0]
        tmp_folder2 = image_path.rsplit('/', 2)[1]
        output_image_path = os.path.join(tmp_folder, "new_model_" + str(count_pig) + "_" + os.path.basename(image_path))
        cv2.imwrite(output_image_path, img)
        return tmp_folder2, mask_pixel_counts

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def natural_sort_key(s):
    """
    Generate a sort key that considers numerical values in strings.
    E.g., "image10.jpg" will be sorted after "image2.jpg".
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]




def main():
    args = parse_arguments()

    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder {args.input_folder} does not exist or is not a directory.")
        exit(1)

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

    results = []  # Step 2: Initialize results list

    # Process each image with a progress bar
    for image_path in tqdm(image_files, desc="Processing Images"):
        # Process the image
        relative_image_path, pixel_sums = process_image(image_path, model, color_palette, args.conf_threshold, args.input_folder)
        
        if relative_image_path is not None:
            sum_result = sum(item['pixel_count'] for item in pixel_sums)
            num_instances = len(pixel_sums)
            results.append({
                'Image Name': relative_image_path,
                'Sum of Pixel Counts': sum_result,
                'Number of Instances': num_instances
            })
            print(f"Processed {relative_image_path}: sum_result={sum_result}, num_instances={num_instances}")
        else:
            print(f"Skipping image {image_path} due to processing error.")
    results.sort(key=lambda x: natural_sort_key(x['Image Name']))

    # Write results to CSV
    csv_file_path = args.output_csv  # You can change the CSV file name or make it an argument
    try:
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['Image Name', 'Sum of Pixel Counts', 'Number of Instances']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for data in results:
                writer.writerow(data)
        print(f"Results have been written to {csv_file_path}")
    except Exception as e:
        print(f"Error writing to CSV file {csv_file_path}: {e}")
        # 处理返回的像素值总和
        # if pixel_sums:
        #     for mask_sum in pixel_sums:
        #         print(f"Instance {mask_sum['instance_id']} pixel sum: {mask_sum['pixel_sum']}")
    print("Processing completed.")

if __name__ == "__main__":
    main()
