import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import argparse
from ultralytics.utils import DEFAULT_CFG, ops

def create_labelme_json(image_path, polygons, output_path, imageHeight, imageWidth):
    
    data = {
        "version": "0.3.3",
        "flags": {},
        "shapes": [],
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth,
    }
    for polygon in polygons:
        #polygon = polygon[0]
        shape = {
            "label": "crack",
            "points": polygon,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        }
        data["shapes"].append(shape)
    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def mask_to_polygons(image_path, overlay_image,imageHeight,imageWidth, output_path, epsilon=0.5):
    """Overlays the perimeter of each mask onto the image at the mask's center.
    
    Args:
        overlay_image: A numpy array of shape (H, W, C), containing masks over a black background.
    
    Returns:
        An image with perimeters annotated at the center of each mask.
    """
    gray_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for contour in contours:
        contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(contour) >= 3:
            polygons.append(contour.reshape(-1, 2).tolist())
    create_labelme_json(image_path, polygons, output_path, imageHeight, imageWidth)  

def overlay_on_black( mask, resize=None):
    """Generates a white mask on a black background based on the given segmentation mask."""
    # Ensure mask is still binary after resizing
    if resize is not None:
        mask = cv2.resize(mask.astype(float), resize)
        mask = mask > 0.5

    # Convert mask to a 3-channel image
    mask_3channel = np.stack([mask]*3, axis=-1) * 255

    return mask_3channel

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument('--input_folder', help='Path to the image files.')
    args = parser.parse_args()
    model = YOLO("/home/deepl/ultralytics_626/712_新loss/weights/best.pt")
    print('Class Names: ', model.names)
    
    image_path_list = [i for i in os.listdir(args.input_folder) if i.endswith(".png") or i.endswith(".jpg")]

    for image_path in image_path_list:
        full_image_path = os.path.join(args.input_folder, image_path)
        print("Processing image: ", full_image_path)

        img = cv2.imread(full_image_path)
        original_h, original_w, _ = img.shape

        # Initialize a black background
        black_background = np.zeros((original_h, original_w, 3), dtype=np.uint8)
        results = model.predict(img, conf=0.25)
        for r in results:
            boxes = r.boxes
            masks = r.masks
        if masks is not None:
            masks = masks.data.cpu().numpy()
            masks = np.moveaxis(masks, 0, -1)
            masks = ops.scale_image( masks, results[0].masks.orig_shape)
            masks = np.moveaxis(masks, -1, 0)
            for seg, box in zip(masks, boxes):
                #seg_resized = cv2.resize(seg, (w, h))
                mask_3channel = overlay_on_black(seg, (original_w, original_h))

                # Apply each mask to the black background
                black_background = np.where(mask_3channel > 0, 255, black_background)

        #cv2.imwrite("result_tmp.png", black_background)
        # replace png or jpg to json in the full_image_path
        output_path = full_image_path.replace(".png", ".json").replace(".jpg", ".json")
        mask_to_polygons(image_path, black_background, original_h, original_w, output_path)