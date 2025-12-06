import cv2


def create_undistorted_maps(K, D, image_size):
    """
    Generate lookup maps to remove image distortion.

    Args:
        K: Camera intrinsic matrix (3x3)
        D: Distortion coefficients
        image_size: Tuple of (height, width) for the image resolution

    Returns:
        map_x, map_y: Lookup maps for cv2.remap() to undistort images
        roi: Region of interest after undistortion (x, y, w, h)
    """
    h, w = image_size

    # Compute optimal camera matrix to handle black borders
    # alpha=0: crop all black pixels; alpha=1: keep all original pixels
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0, newImgSize=(w, h))

    # Generate lookup tables for fast image undistortion
    # CV_16SC2 format is faster and more memory-efficient than CV_32FC1
    map_x, map_y = cv2.initUndistortRectifyMap(
        K,
        D,
        None,  # R (Rotation matrix) - None for monocular cameras
        new_K,  # New camera matrix with optimal parameters
        (w, h),
        cv2.CV_16SC2,
    )

    return map_x, map_y, roi
