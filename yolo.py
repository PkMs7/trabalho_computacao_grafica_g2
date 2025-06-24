import ultralytics
ultralytics.checks()

from ultralytics import YOLO

# A função principal que encapsula o código de execução
def run_yolo_training():
    ultralytics.checks() # É bom manter essa checagem

    model = YOLO('yolo12n.pt')

    # Se você estiver usando um DataLoader com num_workers > 0 (o que é o padrão no Ultralytics para GPUs),
    # essa parte precisa estar protegida pelo if __name__ == '__main__'.
    results = model.train(data="Trabalho_G2_CG.yolov12/data.yaml", epochs=100, imgsz=640, plots=True)


results = model.train(data="Trabalho_G2_CG.yolov12\data.yaml", epochs=100, imgsz=640, plots=True)

