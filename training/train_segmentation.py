import argparse
import logging
import os
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 Segmentation Model")
    parser.add_argument("--model", type=str, default="yolov8m-seg.pt", help="Base model weights")
    parser.add_argument("--data", type=str, default="datasets/f1_segmentation/data.yaml", help="Path to data.yaml")
    parser.add_argument("--project", type=str, default="runs/segment", help="Project output directory")
    parser.add_argument("--name", type=str, default="f1_road_segmentation", help="Experiment name")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    return parser.parse_args()


def train_segmentation_model(
    model_arch: str,
    dataset_yaml: str,
    project_dir: str,
    exp_name: str,
    epochs: int,
    img_size: int,
    batch_size: int,
    patience: int
) -> None:
    """Trains a YOLOv8 instance segmentation model on the provided dataset.

    Args:
        model_arch: The pre-trained YOLO architecture to use as a base.
        dataset_yaml: Relative or absolute path to the data.yaml file.
        project_dir: Directory where Ultralytics stores run artifacts.
        exp_name: The name of the specific training run.
        epochs: Number of training epochs.
        img_size: Target image size for training.
        batch_size: Batch size for training.
        patience: Epochs to wait for improvement before early stopping.
    
    Raises:
        FileNotFoundError: If the dataset YAML file is not found.
    """
    abs_yaml_path = os.path.abspath(dataset_yaml)
    if not os.path.exists(abs_yaml_path):
        raise FileNotFoundError(f"Dataset configuration file not found at: {abs_yaml_path}")

    logger.info("Initializing YOLO segmentation model: %s", model_arch)
    model = YOLO(model_arch)

    logger.info("Starting segmentation training for %d epochs...", epochs)
    try:
        model.train(
            data=abs_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=0,
            project=project_dir,
            name=exp_name,
            patience=patience,
            task="segment",
            exist_ok=True
        )
        logger.info("Training completed successfully.")
        logger.info("Best weights saved to: %s/%s/weights/best.pt", project_dir, exp_name)
    except Exception as e:
        logger.error("Training failed: %s", str(e))
        raise RuntimeError(f"YOLO training encountered an error: {str(e)}") from e


def main() -> None:
    """Main execution entry point."""
    args = parse_arguments()
    try:
        train_segmentation_model(
            model_arch=args.model,
            dataset_yaml=args.data,
            project_dir=args.project,
            exp_name=args.name,
            epochs=args.epochs,
            img_size=args.imgsz,
            batch_size=args.batch,
            patience=args.patience
        )
    except FileNotFoundError as fnf_err:
        logger.error(str(fnf_err))
    except RuntimeError as run_err:
        logger.error(str(run_err))


if __name__ == "__main__":
    main()
