import os
import sys
import numpy as np
import cv2

# Add the root directory to the python path so we can import from run_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_pipeline import unproject_point, get_mask_centroid, is_scene_cut, cull_trails_2d, CULLING_MARGIN

def test_unproject_point():
    """Tests the 3D reprojection math."""
    # Dummy intrinsic matrix K
    k_matrix = np.array([
        [1000.0, 0.0, 500.0],
        [0.0, 1000.0, 500.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Point perfectly in the center of the camera
    u, v, z = 500.0, 500.0, 10.0
    point_3d = unproject_point(u, v, z, k_matrix)
    
    # If the point is at the principal point, X and Y should be 0
    assert np.isclose(point_3d[0], 0.0)
    assert np.isclose(point_3d[1], 0.0)
    assert np.isclose(point_3d[2], 10.0)

    # Point to the right and bottom
    u, v, z = 1000.0, 1000.0, 10.0
    point_3d = unproject_point(u, v, z, k_matrix)
    
    # X = (1000 - 500) * 10 / 1000 = 5.0
    assert np.isclose(point_3d[0], 5.0)
    assert np.isclose(point_3d[1], 5.0)

def test_get_mask_centroid():
    """Tests the centroid calculation of a polygon mask."""
    # Create a simple square polygon
    # Center should be at (10.0, 10.0)
    square_poly = np.array([
        [[0.0, 0.0]],
        [[20.0, 0.0]],
        [[20.0, 20.0]],
        [[0.0, 20.0]]
    ], dtype=np.float32)

    cx, cy = get_mask_centroid(square_poly, fallback_x=99.0, fallback_y=99.0)
    
    assert np.isclose(cx, 10.0)
    assert np.isclose(cy, 10.0)

def test_get_mask_centroid_fallback():
    """Tests if the fallback is triggered when the polygon is empty or invalid."""
    empty_poly = np.array([], dtype=np.float32)
    cx, cy = get_mask_centroid(empty_poly, fallback_x=99.0, fallback_y=99.0)
    
    assert cx == 99.0
    assert cy == 99.0

def test_is_scene_cut():
    """Tests the histogram correlation-based scene cut detector."""
    # Create two identical black images
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img2 = np.zeros((100, 100), dtype=np.uint8)
    
    # Correlation between identical images should be 1.0 (No cut -> False)
    assert not is_scene_cut(img1, img2, threshold=0.5)

    # Create a completely white image
    img3 = np.ones((100, 100), dtype=np.uint8) * 255
    
    # Correlation between black and white images should be 0.0 or negative (Cut -> True)
    assert is_scene_cut(img1, img3, threshold=0.5)

def test_cull_trails_2d():
    """Tests the memory management logic that removes out-of-bounds trail points."""
    width, height = 1920, 1080
    track_history = {
        1: [
            (960.0, 540.0), # Perfectly centered (Keep)
            (-50.0, 0.0), # Slightly out of bounds, but within CULLING_MARGIN (Keep)
            (-5000.0, 540.0), # Way out of bounds horizontally (Remove)
            (960.0, 9999.0) # Way out of bounds vertically (Remove)
        ]
    }
    
    cull_trails_2d(track_history, width, height)
    
    # We expect exactly 2 points to remain
    assert len(track_history[1]) == 2
    assert track_history[1][0] == (960.0, 540.0)
    assert track_history[1][1] == (-50.0, 0.0)
