import ultralytics
ultralytics.checks()

from ultralytics import YOLO

model = YOLO('yolo12n.pt')


results = model.train(data="Trabalho_G2_CG.yolov12\data.yaml", epochs=100, imgsz=640, plots=True)

