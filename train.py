from ultralytics import YOLO

model = YOLO("yolo11s.pt")          # carga peso base COCO
model.train(
    data="configs/cherries_maturity.yaml",
    epochs=120,
    imgsz=800,
    batch=16,
    fraction=1.0,
    project="cherry_yolo11",
    name="maturity_combined",
)
