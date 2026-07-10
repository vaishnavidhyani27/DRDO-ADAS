REAL_OBJECT_HEIGHTS_CM = {
    "Person": 170,
    "Bicycle": 110,
    "Car": 150,
    "Motorcycle": 120,
    "Bus": 300,
    "Truck": 300,
}

FOCAL_LENGTH_PX = 700


def estimate_distance(label, box_height):
    """
    Estimate object distance using its bounding-box height.

    Note:
    This is an approximate distance and should later be calibrated
    using the actual phone camera.
    """

    if box_height <= 0:
        return None

    real_height_cm = REAL_OBJECT_HEIGHTS_CM.get(label)

    if real_height_cm is None:
        return None

    distance_cm = (real_height_cm * FOCAL_LENGTH_PX) / box_height
    distance_m = distance_cm / 100

    return round(distance_m, 1)