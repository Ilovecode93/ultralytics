from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
# model = YOLO('runs/segment/train3/weights/best.pt')
# model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
model = YOLO('yolov8n-seg.pt')
# model = YOLO('checkpoint/520/best.pt')
# Train the model
model.train(data='bigandsmallpig.yaml', epochs=1000, patience= 500, batch=256, cache = True,imgsz=640, device=[0, 1])
