from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("runs/detect/8144/weights/best.pt")

# Export the model to PaddlePaddle format
model.export(format="paddle")  # creates '/yolov8n_paddle_model'

# Load the exported PaddlePaddle model
paddle_model = YOLO("./yolov8n_paddle_model")

# Run inference
results = paddle_model("https://ultralytics.com/images/bus.jpg")