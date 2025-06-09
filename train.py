from ultralytics import YOLO

model = YOLO("yolo11s.pt")          # carga peso base COCO
model.train(
    data="configs/cherries_maturity.yaml",
    epochs=5,
    imgsz=640,
    batch=16,
    fraction=0.2,
    project="cherry_yolo11",
    name="maturity_combined",
)
