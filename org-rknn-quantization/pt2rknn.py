from ultralytics import YOLO
model = YOLO("spacer_1280.pt")
# 强制指定 imgsz=640
model.export(format="rknn", imgsz=1280, opset=12,name="rk3588")