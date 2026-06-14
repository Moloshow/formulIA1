import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Class IDs mapped to BGR colors
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 255),    # F1 car: Red
    1: (150, 250, 50)  # Road: Green/Yellow
}
OPACITY = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate semantic segmentation masks visualizations")
    parser.add_argument("--dataset-dir", type=str, default="datasets/f1_segmentation", help="Segmentation dataset directory")
    return parser.parse_args()


def process_label_file(
    lbl_path: Path,
    width: int,
    height: int
) -> List[Tuple[int, np.ndarray]]:
    """Parses a YOLO segmentation label file and extracts scaled polygons.

    Args:
        lbl_path: Path to the .txt label file.
        width: Image width used for denormalizing coordinates.
        height: Image height used for denormalizing coordinates.

    Returns:
        A list of tuples, each containing the class ID and the polygon points
        formatted for OpenCV drawing functions.
    """
    polygons = []
    if not lbl_path.exists():
        return polygons

    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    pts = []
                    for i in range(0, len(coords), 2):
                        px = int(coords[i] * width)
                        py = int(coords[i + 1] * height)
                        pts.append([px, py])
                    pts_arr = np.array([pts], dtype=np.int32)
                    polygons.append((cls_id, pts_arr))
    except (ValueError, IOError) as e:
        logger.warning("Failed to parse label file %s: %s", lbl_path, str(e))
    
    return polygons


def generate_visualizations(
    dataset_dir: Path,
    colors: Dict[int, Tuple[int, int, int]],
    opacity: float
) -> None:
    """Generates overlaid visualizations and pure masks for the dataset.

    Args:
        dataset_dir: Root directory of the segmentation dataset.
        colors: Mapping of class IDs to BGR colors.
        opacity: Alpha value for the overlay blend.
        
    Raises:
        FileNotFoundError: If the dataset directory does not exist.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for split in ["train", "valid", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            logger.info("Split '%s' not found, skipping.", split)
            continue

        out_overlay_dir = split_dir / "overlays"
        out_mask_dir = split_dir / "masks"
        out_overlay_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        img_paths = list((split_dir / "images").glob("*.*"))
        logger.info("Processing %d images in split '%s'...", len(img_paths), split)

        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Could not read image: %s", img_path)
                continue

            h, w = img.shape[:2]
            mask_rgb = np.zeros_like(img)
            overlay = img.copy()

            lbl_path = split_dir / "labels" / f"{img_path.stem}.txt"
            polygons = process_label_file(lbl_path, w, h)

            for cls_id, pts_arr in polygons:
                color = colors.get(cls_id, (255, 255, 255))
                cv2.fillPoly(mask_rgb, pts_arr, color)
                cv2.fillPoly(overlay, pts_arr, color)

            # Blend the original image with the polygon overlay
            blended = cv2.addWeighted(img, 1.0 - opacity, overlay, opacity, 0)

            # Save the outputs
            cv2.imwrite(str(out_mask_dir / img_path.name), mask_rgb)
            cv2.imwrite(str(out_overlay_dir / img_path.name), blended)

    logger.info("Visualizations generated successfully.")


def main() -> None:
    """Main execution entry point."""
    args = parse_arguments()
    dataset_dir = Path(args.dataset_dir)
    
    logger.info("Starting segmentation dataset visualization generation...")
    try:
        generate_visualizations(
            dataset_dir=dataset_dir,
            colors=CLASS_COLORS,
            opacity=OPACITY
        )
    except FileNotFoundError as fnf_err:
        logger.error(str(fnf_err))
    except Exception as e:
        logger.error("An unexpected error occurred: %s", str(e))


if __name__ == "__main__":
    main()
