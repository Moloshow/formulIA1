import argparse
import logging
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

F1_CAR_CLASS = 0
ROAD_CLASS = 1

TRACKER_CONF = 0.40
TRACKER_IOU = 0.65
EMA_ALPHA = 0.4
MAX_TRAIL_LENGTH = 200
CULLING_MARGIN = 150
FREEZE_MARGIN = 15

TEAM_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Ferrari": (0, 0, 255), "Mercedes": (200, 255, 0), "Red-Bull-Racing": (150, 0, 0),
    "Mclaren": (0, 165, 255), "Alpine": (255, 105, 180), "Aston-Martin": (50, 100, 0),
    "Alfa-Romeo": (50, 0, 150), "Williams": (255, 150, 0), "Alpha-Tauri": (255, 255, 255),
    "Haas": (200, 200, 200)
}

TRACK_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255),
    (0, 255, 127), (255, 192, 203), (255, 69, 0), (147, 112, 219),
    (173, 216, 230), (255, 215, 0), (152, 251, 152), (218, 112, 214)
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="F1 Perception Pipeline and Motion Trails")
    parser.add_argument("--source", type=str, default="videos/sample_f1.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/result.mp4", help="Path to output video")
    parser.add_argument("--weights-seg", type=str, default="runs/segment/f1_road_segmentation/weights/best.pt", help="Path to segmentation model weights")
    parser.add_argument("--weights-cls", type=str, default="runs/classify/f1_team_classifier/weights/best.pt", help="Path to classification model weights")
    parser.add_argument("--classify", action="store_true", help="Enable Team Classifier (Model B)")
    parser.add_argument("--depth-slam", action="store_true", help="Enable 3D SLAM via Depth Anything V2")
    parser.add_argument("--show-mask", action="store_true", help="Display the semantic road mask overlay")
    parser.add_argument("--freeze-edges", action="store_true", help="Freeze tracking near frame edges")
    return parser.parse_args()


def load_models(args: argparse.Namespace, device: str) -> Tuple[YOLO, Optional[YOLO], Any, Any]:
    """Loads necessary neural network models based on CLI arguments."""
    logger.info("Loading YOLO-seg Model (Phase 4) from %s...", args.weights_seg)
    try:
        model_det = YOLO(args.weights_seg)
    except Exception as e:
        logger.error("Failed to load segmentation model: %s", str(e))
        sys.exit(1)

    model_cls = None
    if args.classify:
        logger.info("Loading Classification Model (Phase 1) from %s...", args.weights_cls)
        try:
            model_cls = YOLO(args.weights_cls)
        except Exception as e:
            logger.error("Failed to load classification model: %s", str(e))

    processor_depth = None
    model_depth = None
    if args.depth_slam:
        logger.info("Loading Depth-SLAM models on %s...", device)
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            processor_depth = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
            model_depth = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device)
        except Exception as e:
            logger.error("Failed to load depth models: %s", str(e))

    return model_det, model_cls, processor_depth, model_depth


def unproject_point(u: float, v: float, z: float, k_matrix: np.ndarray) -> np.ndarray:
    """Unprojects a 2D pixel coordinate into 3D camera space."""
    x_3d = (u - k_matrix[0, 2]) * z / k_matrix[0, 0]
    y_3d = (v - k_matrix[1, 2]) * z / k_matrix[1, 1]
    return np.array([x_3d, y_3d, z], dtype=np.float32)


def is_scene_cut(prev_gray: np.ndarray, curr_gray: np.ndarray, threshold: float = 0.5) -> bool:
    """Detects camera cuts using histogram correlation.
    
    Optical flow is highly sensitive to motion blur during whip pans. 
    Histogram correlation ignores spatial layout and evaluates overall color 
    distribution, making it robust against blur while highly sensitive to actual cuts.
    """
    hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
    hist_curr = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
    correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
    return correlation < threshold


def apply_homography_to_history(
    track_history: Dict[int, List[Tuple[float, float]]],
    homography_matrix: np.ndarray
) -> None:
    """Applies a 2D Homography transformation to all historical trail points."""
    for t_id, pts in track_history.items():
        if not pts:
            continue
        pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts_arr, homography_matrix)
        
        track_history[t_id] = [
            (float(p[0][0]), float(p[0][1])) for p in transformed
        ]


def apply_pnp_to_history(
    track_history: Dict[int, List[np.ndarray]],
    rvec: np.ndarray,
    tvec: np.ndarray
) -> None:
    """Applies a 3D Rigid Body Transformation to historical points."""
    r_matrix, _ = cv2.Rodrigues(rvec)
    for t_id, pts in track_history.items():
        if not pts:
            continue
        pts_arr = np.array(pts).reshape(-1, 3).T
        pts_new = (r_matrix @ pts_arr) + tvec
        track_history[t_id] = [pts_new[:, j] for j in range(pts_new.shape[1])]


def get_mask_centroid(mask_poly: np.ndarray, fallback_x: float, fallback_y: float) -> Tuple[float, float]:
    """Calculates the physical center of mass of a segmentation mask.
    
    Using the mask centroid instead of the bounding box center prevents 
    trajectory jitter caused by occlusion or bounding box truncation.
    """
    if len(mask_poly) > 0:
        moments = cv2.moments(mask_poly)
        if moments["m00"] != 0:
            return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
    return fallback_x, fallback_y


def cull_trails_2d(track_history: Dict[int, List[Tuple[float, float]]], width: int, height: int) -> None:
    """Removes points that have drifted far beyond the visible frustum to save memory."""
    for t_id in list(track_history.keys()):
        valid_pts = [
            p for p in track_history[t_id]
            if -CULLING_MARGIN <= p[0] <= width + CULLING_MARGIN
            and -CULLING_MARGIN <= p[1] <= height + CULLING_MARGIN
        ]
        track_history[t_id] = valid_pts[-MAX_TRAIL_LENGTH:]


# -----------------------------------------------------------------------------
# Main Application Class
# -----------------------------------------------------------------------------
class F1Pipeline:
    """Orchestrates the computer vision pipeline for F1 analysis."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_det, self.model_cls, self.processor_depth, self.model_depth = load_models(args, self.device)
        
        self.track_history: Dict[int, List[Any]] = defaultdict(list)
        self.track_classes: Dict[int, str] = {}
        self.track_colors_dict: Dict[int, Tuple[int, int, int]] = {}
        
        self.cap = cv2.VideoCapture(self.args.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source at {self.args.source}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.args.output, fourcc, self.fps, (self.width, self.height))

        # Camera intrinsic approximation
        focal_length = self.width
        self.k_matrix = np.array(
            [[focal_length, 0, self.width / 2],
             [0, focal_length, self.height / 2],
             [0, 0, 1]], dtype=np.float32
        )
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        self.prev_gray = None
        self.prev_depth = None
        self.prev_boxes = []
        self.prev_road_mask = None

    def _estimate_ego_motion(self, curr_gray: np.ndarray, depth_map: Optional[np.ndarray]) -> None:
        """Calculates camera movement and updates historical points accordingly."""
        if self.prev_gray is None:
            return

        # Restrict tracking to the flat, static road surface to obtain true camera motion
        mask = self.prev_road_mask if self.prev_road_mask is not None else np.ones_like(self.prev_gray) * 255
        
        if self.prev_road_mask is None:
            # Fallback geometric mask to avoid HUD and UI elements
            m_top, m_bot = int(self.height * 0.15), int(self.height * 0.85)
            m_left, m_right = int(self.width * 0.15), int(self.width * 0.85)
            cv2.rectangle(mask, (0, 0), (self.width, m_top), 0, -1)
            cv2.rectangle(mask, (0, m_bot), (self.width, self.height), 0, -1)
            cv2.rectangle(mask, (0, 0), (m_left, self.height), 0, -1)
            cv2.rectangle(mask, (m_right, 0), (self.width, self.height), 0, -1)
            for box in self.prev_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(mask, (max(0, x1 - 50), max(0, y1 - 50)),
                              (min(self.width, x2 + 50), min(self.height, y2 + 50)), 0, -1)

        p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=30, mask=mask)
        if p0 is None:
            return

        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, p0, None, winSize=(31, 31), maxLevel=3)
        if p1 is None or st is None:
            return

        if is_scene_cut(self.prev_gray, curr_gray):
            self.track_history.clear()
            return

        if self.args.depth_slam and self.prev_depth is not None:
            self._apply_depth_slam(p0, p1, st)
        else:
            self._apply_homography(p0, p1, st)

    def _apply_depth_slam(self, p0: np.ndarray, p1: np.ndarray, st: np.ndarray) -> None:
        """3D reprojection using Depth Anything V2 and Perspective-n-Point."""
        p0_3d, p1_2d = [], []
        for i in range(len(p0)):
            if st[i] == 1:
                u, v = int(p0[i, 0, 0]), int(p0[i, 0, 1])
                u, v = np.clip(u, 0, self.width - 1), np.clip(v, 0, self.height - 1)
                z = 100.0 / max(1e-3, self.prev_depth[v, u])
                p0_3d.append(unproject_point(u, v, z, self.k_matrix))
                p1_2d.append(p1[i, 0])
        
        if len(p0_3d) >= 6:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(p0_3d, dtype=np.float32),
                np.array(p1_2d, dtype=np.float32),
                self.k_matrix, self.dist_coeffs,
                reprojectionError=5.0, flags=cv2.SOLVEPNP_EPNP
            )
            # Physical sanity check: reject impossible camera leaps
            if success and inliers is not None and len(inliers) >= 4:
                if np.linalg.norm(tvec) < 200.0 and np.linalg.norm(rvec) < 0.5:
                    apply_pnp_to_history(self.track_history, rvec, tvec)

    def _apply_homography(self, p0: np.ndarray, p1: np.ndarray, st: np.ndarray) -> None:
        """2D planar reprojection for road surface tracking."""
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        if len(good_new) >= 4:
            h_matrix, inliers = cv2.findHomography(good_old, good_new, cv2.RANSAC, 3.0)
            if h_matrix is not None and inliers is not None and np.sum(inliers) >= 4:
                det = np.linalg.det(h_matrix[0:2, 0:2])
                if 0.5 < det < 2.0:
                    apply_homography_to_history(self.track_history, h_matrix)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Main processing loop for a single frame."""
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_map = None

        if self.args.depth_slam:
            inputs = self.processor_depth(images=frame, return_tensors="pt").to(self.device)
            with torch.no_grad():
                depth_output = self.model_depth(**inputs).predicted_depth
                depth_map = torch.nn.functional.interpolate(
                    depth_output.unsqueeze(1), size=(self.height, self.width),
                    mode="bicubic", align_corners=False
                ).squeeze().cpu().numpy()

        self._estimate_ego_motion(curr_gray, depth_map)

        self.prev_gray = curr_gray
        self.prev_depth = depth_map

        # Execute Object Detection & Segmentation
        results = self.model_det.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=[F1_CAR_CLASS, ROAD_CLASS],
            conf=TRACKER_CONF, iou=TRACKER_IOU, retina_masks=True,
            imgsz=1280, device=0, verbose=False
        )

        annotated_frame = frame.copy()
        overlay = annotated_frame.copy()

        # Update semantic road mask
        curr_road_mask = None
        if results[0].masks is not None:
            curr_road_mask = np.zeros_like(curr_gray)
            for i, cls_id in enumerate(results[0].boxes.cls.int().cpu().tolist()):
                if cls_id == ROAD_CLASS:
                    pts = np.array([results[0].masks.xyn[i] * [self.width, self.height]], dtype=np.int32)
                    cv2.fillPoly(curr_road_mask, pts, 255)
            
            # Punch holes in the mask for the cars
            for i, cls_id in enumerate(results[0].boxes.cls.int().cpu().tolist()):
                if cls_id == F1_CAR_CLASS:
                    x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i])
                    cv2.rectangle(
                        curr_road_mask,
                        (max(0, x1 - 50), max(0, y1 - 50)),
                        (min(self.width, x2 + 50), min(self.height, y2 + 50)),
                        0, -1
                    )
            
            self.prev_road_mask = curr_road_mask
            
            if self.args.show_mask:
                green_mask = np.zeros_like(annotated_frame)
                green_mask[:, :, 1] = curr_road_mask
                annotated_frame = cv2.addWeighted(annotated_frame, 1.0, green_mask, 0.3, 0)
        else:
            self.prev_road_mask = None

        self._update_trajectories(results, annotated_frame, depth_map)
        self._draw_trails(overlay)

        # Blend trails
        final_frame = cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0)
        return final_frame

    def _update_trajectories(self, results: Any, annotated_frame: np.ndarray, depth_map: Optional[np.ndarray]) -> None:
        """Updates and draws real-time bounding boxes and trajectories."""
        if results[0].boxes.id is None:
            self.prev_boxes = []
            return

        boxes_xyxy = results[0].boxes.xyxy.cpu()
        boxes_xywh = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cls_ids = results[0].boxes.cls.int().cpu().tolist()
        self.prev_boxes = []

        for i, track_id in enumerate(track_ids):
            if cls_ids[i] == ROAD_CLASS:
                continue

            x1, y1, x2, y2 = map(int, boxes_xyxy[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.width, x2), min(self.height, y2)
            self.prev_boxes.append([x1, y1, x2, y2])
            
            team_name = "Formula 1"
            if self.args.classify:
                if track_id in self.track_classes:
                    team_name = self.track_classes[track_id]
                elif x2 > x1 and y2 > y1:
                    crop = annotated_frame[y1:y2, x1:x2]
                    cls_results = self.model_cls(crop, verbose=False)
                    team_name = cls_results[0].names[cls_results[0].probs.top1]
                    self.track_classes[track_id] = team_name
                color = TEAM_COLORS.get(team_name, (0, 255, 255))
            else:
                color = TRACK_COLORS[track_id % len(TRACK_COLORS)]

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, team_name, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            self.track_colors_dict[track_id] = color

            x_ctr, y_ctr, _, _ = boxes_xywh[i]
            is_truncated = self.args.freeze_edges and (
                x1 < FREEZE_MARGIN or y1 < FREEZE_MARGIN or
                x2 > self.width - FREEZE_MARGIN or y2 > self.height - FREEZE_MARGIN
            )

            if not is_truncated:
                mask_poly = results[0].masks.xy[i] if results[0].masks is not None else []
                cx, cy = get_mask_centroid(mask_poly, float(x_ctr), float(y_ctr))
                
                if self.args.depth_slam:
                    u, v = np.clip(int(cx), 0, self.width - 1), np.clip(int(cy), 0, self.height - 1)
                    z_car = 100.0 / max(1e-3, depth_map[v, u])
                    current_pt = unproject_point(u, v, z_car, self.k_matrix)
                else:
                    current_pt = np.array([cx, cy])

                if len(self.track_history[track_id]) > 0:
                    prev_pt = np.array(self.track_history[track_id][-1])
                    smoothed_pt = EMA_ALPHA * current_pt + (1.0 - EMA_ALPHA) * prev_pt
                else:
                    smoothed_pt = current_pt
                    
                if self.args.depth_slam:
                    self.track_history[track_id].append(smoothed_pt)
                else:
                    self.track_history[track_id].append((float(smoothed_pt[0]), float(smoothed_pt[1])))

    def _draw_trails(self, overlay: np.ndarray) -> None:
        """Draws the motion trails onto the overlay layer."""
        if not self.args.depth_slam:
            cull_trails_2d(self.track_history, self.width, self.height)

        for t_id, pts in self.track_history.items():
            color = self.track_colors_dict.get(t_id, (0, 255, 255))
            
            if self.args.depth_slam:
                if len(pts) > MAX_TRAIL_LENGTH:
                    self.track_history[t_id] = pts[-MAX_TRAIL_LENGTH:]
                pts = self.track_history[t_id]
                if len(pts) > 1:
                    pts_2d, _ = cv2.projectPoints(
                        np.array(pts, dtype=np.float32), np.zeros((3, 1)), np.zeros((3, 1)),
                        self.k_matrix, self.dist_coeffs
                    )
                    pts_2d = pts_2d.reshape(-1, 2)
                    for j in range(1, len(pts_2d)):
                        p1_draw = (int(pts_2d[j - 1][0]), int(pts_2d[j - 1][1]))
                        p2_draw = (int(pts_2d[j][0]), int(pts_2d[j][1]))
                        thickness = int(max(1, 8 * (j / len(pts_2d))))
                        cv2.line(overlay, p1_draw, p2_draw, color, thickness, cv2.LINE_AA)
            else:
                if len(pts) > 1:
                    for j in range(1, len(pts)):
                        p1_draw = (int(pts[j - 1][0]), int(pts[j - 1][1]))
                        p2_draw = (int(pts[j][0]), int(pts[j][1]))
                        thickness = int(max(1, 8 * (j / len(pts))))
                        cv2.line(overlay, p1_draw, p2_draw, color, thickness, cv2.LINE_AA)

    def run(self) -> None:
        """Executes the pipeline on the video source."""
        logger.info("Processing %dx%d @ %d FPS", self.width, self.height, self.fps)

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                processed_frame = self._process_frame(frame)
                
                self.out.write(processed_frame)
                cv2.imshow("FormulIA1 - Vision Pipeline", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            logger.error("Pipeline encountered a fatal error: %s", str(e))
        finally:
            self.cap.release()
            self.out.release()
            cv2.destroyAllWindows()
            logger.info("Pipeline execution completed.")


def main() -> None:
    """Main execution entry point."""
    args = parse_arguments()
    try:
        pipeline = F1Pipeline(args)
        pipeline.run()
    except Exception as e:
        logger.error("Failed to initialize pipeline: %s", str(e))


if __name__ == "__main__":
    main()
