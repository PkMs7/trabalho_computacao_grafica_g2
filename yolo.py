import ultralytics
from ultralytics import YOLO

# A função principal que encapsula o código de execução
def run_yolo_training():
    ultralytics.checks() # É bom manter essa checagem

    model = YOLO('yolo12m.pt')

    # Se você estiver usando um DataLoader com num_workers > 0 (o que é o padrão no Ultralytics para GPUs),
    # essa parte precisa estar protegida pelo if __name__ == '__main__'.
    results = model.train(data="Trabalho_G2_CG.yolov12/data.yaml", epochs=200, imgsz=640, plots=True)

    # Opcional: Você pode adicionar mais código aqui para pós-processamento, etc.
    print("Treinamento concluído!")

# Este é o bloco de proteção crucial para Windows
if __name__ == '__main__':
    # No Windows, se você planeja "congelar" seu programa em um executável (ex: com PyInstaller),
    # você também precisaria da linha abaixo. Para scripts Python normais, não é estritamente necessário,
    # mas não faz mal em manter.
    # from multiprocessing import freeze_support
    # freeze_support()

    run_yolo_training()