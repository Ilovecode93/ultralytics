from ultralytics import YOLO
from PIL import Image
import os

# Load a model
model = YOLO("/home/deepl/ultralytics_626/712_新loss/weights/best.pt")  # pretrained YOLOv8n model
# Run batched inference on a list of images
results = model("/media/deepl/8206e4e5-7eb6-48c5-969a-e61fbc5adff7/Bigpig/923")  # return a list of Results objects
# 创建单个目录
directory = "/media/deepl/8206e4e5-7eb6-48c5-969a-e61fbc5adff7/Bigpig/results_923"

if not os.path.exists(directory):
    os.mkdir(directory)
    print(f"Directory '{directory}' created")



# Process results list
# for result in results:
#     print("result: ", result)
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     # result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk
# Visualize the results

for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot(conf = False, font_size = 1, line_width = 1, labels = False,probs = False)  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    save_path = os.path.join(directory, f"result_{i}.jpg")
    im_rgb.save(save_path)
    print(f"Saved image: {save_path}")
    
    
    
    # Show results to screen (in supported environments)
    # r.show()
    #r.save(filename=f"/media/deepl/8206e4e5-7eb6-48c5-969a-e61fbc5adff7/Bigpig/results_9_22_pig/results{i}.jpg")