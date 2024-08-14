from ultralytics import YOLO
import cv2
from ultralytics.utils import LOGGER, NUM_THREADS, ops
import numpy as np
import os

def predict_on_image(model, img, conf = 0.1):
    result = model(img, conf=conf)[0]

    # detection
    # result.boxes.xyxy   # box with xyxy format, (N, 4)
    cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
    probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
    boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)
    if len(boxes) == 0:
        return [], [], [], []
    # segmentation
    masks = result.masks.data.cpu().numpy()     # masks, (N, H, W)

    masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
    # rescale masks to original image
    masks = ops.scale_image(masks, masks.shape[:2], result.masks.orig_shape)
    masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)

    return boxes, masks, cls, probs


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    # colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    # colored_mask = np.moveaxis(colored_mask, 0, -1)
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Ensure mask has 3 channels
    colored_mask = np.stack([mask_resized, mask_resized, mask_resized], axis=-1) 
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    
    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)
    
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    
    return image_combined


# Load a model
model = YOLO('/home/deepl/ultralytics_626/626日训练/segment/train/weights/best.pt')
input_folder = "/home/deepl/Grounded-Segment-Anything/61_images"
output_folder = "/home/deepl/ultralytics/output2/"
def preprocess(img_nparray):
    pts = np.array([(535, 494), (532, 777), (573, 1005), (662, 1272), 
                    (721, 1392), (826, 1553), (1005, 1539), (1164, 1498), (1288, 1443),(1364, 1400),
                    (1373, 1226),(1377, 1011),(1333,806),(1297, 707),(1219,544),(1169,461),(1091,364),(802,395)], np.int32)
    new_mask = np.zeros_like(img_nparray[:, :, 0])
    cv2.fillPoly(new_mask, [pts], 255)
    masked_img = cv2.bitwise_and(img, img, mask=new_mask)
    
    return masked_img
    

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for i in os.listdir(input_folder):
    full_path = os.path.join(input_folder, i)
    img = cv2.imread(full_path)
    after_img = preprocess(img)
    boxes, masks, cls, probs = predict_on_image(model, after_img)
    print("len boxes: ", len(boxes))
    if len(boxes) == 0:
        cv2.imwrite(os.path.join(output_folder, i), img)
        continue
    image_with_masks = np.copy(after_img)
    
    for mask_i in masks:
        image_with_masks = overlay(image_with_masks, mask_i, color=(0,255,0), alpha=0.3)
    cv2.imwrite(os.path.join(output_folder, i), image_with_masks)

