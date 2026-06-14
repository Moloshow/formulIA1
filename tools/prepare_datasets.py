import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any

import cv2
import yaml

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare detection and classification datasets")
    parser.add_argument("--base-dir", type=str, default="datasets/f1_generic", help="Source generic dataset")
    parser.add_argument("--det-dir", type=str, default="datasets/f1_detection", help="Target detection dataset")
    parser.add_argument("--cls-dir", type=str, default="datasets/f1_classification", help="Target classification dataset")
    return parser.parse_args()


def read_dataset_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Reads the dataset configuration YAML file.

    Args:
        yaml_path: Path to the data.yaml file.

    Returns:
        The parsed YAML data as a dictionary.

    Raises:
        FileNotFoundError: If the YAML file is missing.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_detection_yaml(output_dir: Path) -> None:
    """Creates a new data.yaml specifically for single-class object detection.

    Args:
        output_dir: The directory where the new dataset will reside.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    det_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": 1,
        "names": ["f1_car"]
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(det_yaml, f)
    logger.info("Created single-class detection YAML at %s", output_dir)


def process_dataset_split(
    split: str,
    class_names: List[str],
    base_dir: Path,
    det_dir: Path,
    cls_dir: Path
) -> None:
    """Processes a dataset split to separate detection and classification data.

    Args:
        split: Split name ('train', 'valid', 'test').
        class_names: List of class names from the original dataset.
        base_dir: Original dataset directory.
        det_dir: Target directory for detection data.
        cls_dir: Target directory for classification crops.
    """
    img_dir = base_dir / split / "images"
    lbl_dir = base_dir / split / "labels"

    if not img_dir.exists():
        logger.warning("Directory %s does not exist, skipping split.", img_dir)
        return

    out_det_img = det_dir / split / "images"
    out_det_lbl = det_dir / split / "labels"
    out_det_img.mkdir(parents=True, exist_ok=True)
    out_det_lbl.mkdir(parents=True, exist_ok=True)

    for cname in class_names:
        (cls_dir / split / cname).mkdir(parents=True, exist_ok=True)

    img_paths = list(img_dir.glob("*.*"))
    logger.info("Processing %d images in split '%s'...", len(img_paths), split)

    for img_path in img_paths:
        if img_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read image %s", img_path)
            continue
            
        height, width = img.shape[:2]
        
        # Copy image to the generic detection dataset
        shutil.copy(img_path, out_det_img / img_path.name)

        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_det_lines = []
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            if cls_id >= len(class_names):
                continue
                
            c_name = class_names[cls_id]
            x_center, y_center, bbox_w, bbox_h = map(float, parts[1:5])

            # Extract image crops for the image classification model (Cascade Model B)
            x1 = int((x_center - bbox_w / 2) * width)
            y1 = int((y_center - bbox_h / 2) * height)
            x2 = int((x_center + bbox_w / 2) * width)
            y2 = int((y_center + bbox_h / 2) * height)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                crop_path = cls_dir / split / c_name / f"{img_path.stem}_{i}.jpg"
                cv2.imwrite(str(crop_path), crop)

            # Rewrite label to generic 'f1_car' class (ID 0) for detection (Cascade Model A)
            parts[0] = "0"
            new_det_lines.append(" ".join(parts))

        with open(out_det_lbl / lbl_path.name, "w", encoding="utf-8") as f:
            f.write("\n".join(new_det_lines) + "\n")


def main() -> None:
    """Main execution entry point."""
    args = parse_arguments()
    base_dir = Path(args.base_dir)
    det_dir = Path(args.det_dir)
    cls_dir = Path(args.cls_dir)

    logger.info("Starting preparation of Two-Stage cascade datasets...")
    
    try:
        yaml_data = read_dataset_yaml(base_dir / "data.yaml")
        class_names = yaml_data.get("names", [])
        
        if not class_names:
            logger.error("No class names found in data.yaml")
            return
            
        logger.info("Discovered original classes: %s", class_names)
        
        setup_detection_yaml(det_dir)

        for split in ["train", "valid", "test"]:
            process_dataset_split(
                split=split,
                class_names=class_names,
                base_dir=base_dir,
                det_dir=det_dir,
                cls_dir=cls_dir
            )
            
        logger.info("Successfully isolated detection and classification datasets.")
        
    except Exception as e:
        logger.error("Data preparation failed: %s", str(e))


if __name__ == "__main__":
    main()
