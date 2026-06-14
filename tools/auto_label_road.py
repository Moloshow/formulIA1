import argparse
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import SAM

# -----------------------------------------------------------------------------
# Default Configuration (Overridable via CLI)
# -----------------------------------------------------------------------------
SEGFORMER_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
SAM_WEIGHTS = "sam_b.pt"
GROUND_CLASSES = [0, 1, 9]
NEW_F1_CLASS = 0
NEW_ROAD_CLASS = 1
MIN_CONTOUR_AREA = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label road and F1 cars using SAM and Segformer")
    parser.add_argument("--input-dir", type=str, default="datasets/f1_detection", help="Source detection dataset directory")
    parser.add_argument("--output-dir", type=str, default="datasets/f1_segmentation", help="Target segmentation dataset directory")
    return parser.parse_args()


def init_models(device: str) -> tuple:
    """Initializes the SegFormer and SAM models.

    Args:
        device: The computing device ('cuda' or 'cpu').

    Returns:
        A tuple containing the SegFormer processor, the SegFormer model, and the SAM model.
    """
    logger.info("Loading SegFormer for road extraction...")
    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_ID)
    segformer_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_ID).to(device)

    logger.info("Loading SAM for F1 pixel-perfect segmentation...")
    sam_model = SAM(SAM_WEIGHTS)

    return processor, segformer_model, sam_model


def process_dataset_split(
    split: str,
    input_dir: Path,
    output_dir: Path,
    processor: SegformerImageProcessor,
    segformer_model: SegformerForSemanticSegmentation,
    sam_model: SAM,
    device: str
) -> None:
    """Processes a single dataset split to auto-label road and car masks.

    Args:
        split: The dataset split name (e.g., 'train', 'valid', 'test').
        input_dir: Source dataset directory.
        output_dir: Target dataset directory.
        processor: The Segformer image processor.
        segformer_model: The Segformer model instance.
        sam_model: The SAM model instance.
        device: The computing device.
    """
    img_dir = input_dir / split / "images"
    lbl_dir = input_dir / split / "labels"

    img_paths = list(img_dir.glob("*.*"))
    if not img_paths:
        return

    out_img_dir = output_dir / split / "images"
    out_lbl_dir = output_dir / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing '%s' split (%d images)...", split, len(img_paths))

    for img_path in img_paths:
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to open image %s: %s", img_path, str(e))
            continue

        width, height = image.size
        new_labels = []

        # 1. Extract road mask using SegFormer
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = segformer_model(**inputs)
            logits = outputs.logits

        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        road_mask = np.isin(pred_seg, GROUND_CLASSES).astype(np.uint8) * 255

        # 2. Convert mask to simplified polygons
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Filter out microscopic noise patches
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                poly_str = f"{NEW_ROAD_CLASS}"
                for point in approx:
                    x, y = point[0]
                    poly_str += f" {x/width:.5f} {y/height:.5f}"
                new_labels.append(poly_str)

        # 3. Refine F1 bounding boxes into perfect masks using SAM
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            bboxes_for_sam = []
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = (cx - w / 2) * width
                        y1 = (cy - h / 2) * height
                        x2 = (cx + w / 2) * width
                        y2 = (cy + h / 2) * height
                        bboxes_for_sam.append([x1, y1, x2, y2])
            
            if bboxes_for_sam:
                image_np = cv2.imread(str(img_path))
                sam_results = sam_model(image_np, bboxes=bboxes_for_sam, verbose=False)
                
                if sam_results[0].masks is not None:
                    for mask_poly in sam_results[0].masks.xyn:
                        poly_str = f"{NEW_F1_CLASS}"
                        for point in mask_poly:
                            poly_str += f" {point[0]:.5f} {point[1]:.5f}"
                        new_labels.append(poly_str)

        # Save the new semantic labels and copy the image
        with open(out_lbl_dir / f"{img_path.stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(new_labels))
            
        shutil.copy(img_path, out_img_dir / img_path.name)


def generate_yaml_config(output_dir: Path) -> None:
    """Generates the required data.yaml file for YOLOv8-seg training."""
    yaml_data = {
        "path": str(output_dir.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": ["f1_car", "road"]
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    logger.info("Generated dataset configuration at %s/data.yaml", output_dir)


def main() -> None:
    """Main execution entry point."""
    args = parse_arguments()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error("Input directory %s not found. Please ensure the detection dataset exists.", input_dir)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device.upper())

    try:
        processor, segformer_model, sam_model = init_models(device)

        for split in ["train", "valid", "test"]:
            process_dataset_split(split, input_dir, output_dir, processor, segformer_model, sam_model, device)
            
        generate_yaml_config(output_dir)
        logger.info("Auto-labeling process completed successfully.")
        
    except Exception as e:
        logger.error("An unexpected error occurred during auto-labeling: %s", str(e))


if __name__ == "__main__":
    main()
