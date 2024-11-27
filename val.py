from ultralytics import YOLO
#model=YOLO("checkpoint/625/weights/best.pt")
#model625 = model.val(data='coco128-seg.yaml')
#print("65 model map: ", model625.box.map)
#del model
model=YOLO("/home/deepl/ultralytics_626/712_新loss/weights/best.pt")
model712 = model.val(data='ocr.yaml')
print("label model map: ",model712.box.map)
del model
