import argparse
import logging
import os
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Model")
    parser.add_argument("--model", type=str, default="runs/segment/f1_road_segmentation/weights/best.pt", help="Path to trained model weights")
    parser.add_argument("--data", type=str, default="datasets/f1_segmentation/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size used for evaluation")
    return parser.parse_args()


def evaluate_segmentation_model(
    model_path: str,
    dataset_yaml: str,
    split: str,
    img_size: int
) -> None:
    """Evaluates a trained YOLOv8 model on a designated dataset split.

    Args:
        model_path: Path to the trained YOLO weights (.pt).
        dataset_yaml: Path to the data configuration YAML file.
        split: The dataset split to evaluate on (e.g., 'test', 'val').
        img_size: Image size used during evaluation.

    Raises:
        FileNotFoundError: If the model weights or dataset YAML cannot be found.
    """
    abs_model_path = os.path.abspath(model_path)
    abs_yaml_path = os.path.abspath(dataset_yaml)

    if not os.path.exists(abs_model_path):
        raise FileNotFoundError(f"Model weights not found at: {abs_model_path}")
    if not os.path.exists(abs_yaml_path):
        raise FileNotFoundError(f"Dataset configuration file not found at: {abs_yaml_path}")

    logger.info("Loading YOLO segmentation model from: %s", abs_model_path)
    model = YOLO(abs_model_path)

    logger.info("Evaluating model on the '%s' split...", split)
    try:
        metrics = model.val(
            data=abs_yaml_path,
            split=split,
            device=0,
            imgsz=img_size
        )
        logger.info("Evaluation completed successfully.")
        logger.info("Mean Average Precision (mAP50): %.3f", metrics.box.map50)
        logger.info("Global Precision (mAP50-95): %.3f", metrics.box.map)
    except Exception as e:
        logger.error("Evaluation failed: %s", str(e))
        raise RuntimeError(f"YOLO evaluation encountered an error: {str(e)}") from e


def main() -> None:
    """Main execution entry point."""
    args = parse_arguments()
    try:
        evaluate_segmentation_model(
            model_path=args.model,
            dataset_yaml=args.data,
            split=args.split,
            img_size=args.imgsz
        )
    except FileNotFoundError as fnf_err:
        logger.error(str(fnf_err))
    except RuntimeError as run_err:
        logger.error(str(run_err))


if __name__ == "__main__":
    main()
