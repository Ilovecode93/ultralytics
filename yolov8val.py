from ultralytics import YOLO
#model=YOLO("checkpoint/625/weights/best.pt")
#model625 = model.val(data='coco128-seg.yaml')
#print("65 model map: ", model625.box.map)
#del model
# model=YOLO("/home/deepl/usedlocated_ultralytics/smallpig_weights/1110/weights/best.pt")
# model1110 = model.val(data='smallpig.yaml')
# print("1110 model map: ",model1110.box.map)
# del model
# #model=YOLO("checkpoint/82/weights/best.pt")
# #model82 = model.val(data='coco128-seg.yaml')
# #print("model82 model map: ",model82.box.map)
# #del model
# model=YOLO("/home/deepl/usedlocated_ultralytics/smallpig_weights/1029/weights/best.pt")
# model1029 = model.val(data='smallpig.yaml')
# print("1029 model map: ",model1029.box.map)
# del model
# model = YOLO("/home/deepl/usedlocated_ultralytics/smallpig_weights/1115/weights/best.pt")
# model1115 = model.val(data='smallpig.yaml')
# print("1115 model map: ",model1115.box.map)
# del model
# model = YOLO("/home/deepl/usedlocated_ultralytics/smallpig_weights/1117/weights/best.pt")
# model1117 = model.val(data='smallpig.yaml')
# print("1117 model map: ",model1117.box.map)
#del model
model = YOLO("/home/deepl/usedlocated_ultralytics/smallpig_weights/1128/weights/best.pt")
model1128 = model.val(data='val.yaml')
print("去年训练的模型 1128 model map: ",model1128.box.map)
del model
# model = YOLO("/home/deepl/ultralytics_626/626日训练/segment/train/weights/best.pt")
# model628 = model.val(data='val.yaml')
# print("628 model map: ",model628.box.map)
# del model
model = YOLO("/home/deepl/ultralytics_626/7_3训练/weights/best.pt")
model73 = model.val(data='val.yaml')
print("73 model map: ",model73.box.map)
del model
# model = YOLO("/home/deepl/ultralytics_626/7_7训练/weights/best.pt")
# model77 = model.val(data='val.yaml')
# print("77 model map: ",model77.box.map)
# del model
model = YOLO("/home/deepl/ultralytics/712/weights/best.pt")
model712 = model.val(data='val.yaml')
print("712 model map: ",model712.box.map)
del model
