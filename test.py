# from ultralytics import YOLO
# from pathlib import Path

# model = YOLO(r"C:\Users\camil\OneDrive\Escritorio\Nueva carpeta\cherry_co_model\cherry_yolo11\maturity_combined2\weights\best.pt")

# results = model(r"C:\Users\camil\OneDrive\Escritorio\Nueva carpeta\cherry_co_model\dataset_ripeness\val\images\cherry_N00299_2021_12_21_H08_50_ripe_fruits_visit2_plot1_row_01.JPG", conf=0.25, save=True)

# img_dir = Path("data/pruebas")
# for img in img_dir.glob("*.jpg"):
#     results = model(img, save=True)

# for r in results:
#     for box in r.boxes:
#         cls = int(box.cls)         
#         conf = float(box.conf)
#         x1, y1, x2, y2 = box.xyxy[0] 

#!/usr/bin/env python3
"""
 test.py - Inference script for your trained Cherry Ripeness YOLO model.

 Example usage:
     python test.py --weights runs/detect/ripeness_v1/weights/best.pt \
                    --source path/to/image.jpg \
                    --imgsz 640 --conf 0.25 --save

 This will run inference on the given image and, if --save is passed, store
 an annotated copy in runs/test/.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_opt():
    """Parse command‑line options."""
    parser = argparse.ArgumentParser(description="Run YOLO inference on a single image")
    parser.add_argument(
        "--weights", type=str, default="best.pt", help="Path to trained .pt model"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to an image file on which to run inference",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size (pixels)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold"
    )
    parser.add_argument(
        "--save", action="store_true", help="Save annotated image to runs/test/"
    )
    return parser.parse_args()


def main(opt):
    """Load model and run prediction."""
    model = YOLO(opt.weights)

    # Run prediction
    results = model.predict(
        opt.source,
        imgsz=opt.imgsz,
        conf=opt.conf,
        save=opt.save,
        project="runs",
        name="test",
    )

    # Print a short, human‑readable summary of results
    for r in results:
        print(r.verbose())  # e.g. "image.jpg: 640x480 1 green, 2 halfripe, 3 ripe"
        # You can also inspect r.boxes.xyxy, r.names, etc.

    if opt.save:
        save_dir = Path(results[0].save_dir)
        print(f"Annotated image saved to {save_dir / Path(opt.source).name}")


if __name__ == "__main__":
    opts = parse_opt()
    main(opts)


# python test.py --weights runs/detect/ripeness_v1/weights/best.pt \
#                --source pruebas/mi_foto.jpg \
#                --save