# from ultralytics import YOLO

# model = YOLO("yolov8n-seg.pt")
# model.train(data="smalldataset.yaml", name = "712", epochs=100, batch=16, cache = True)

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="/home/deepl/OCR_LABEL/ocr.v3i.yolov8/data.yaml", name = "1030", epochs=2, batch=16, cache = True)
