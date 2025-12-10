# pose_estimator.py

class PoseEstimator:
    def __init__(self, intrinsics: dict):
        """
        intrinsics 예:
          { "fx": ..., "fy": ..., "ppx": ..., "ppy": ... }
        또는
          { "fx": ..., "fy": ..., "cx":  ..., "cy":  ... }
        """
        self.fx = intrinsics["fx"]
        self.fy = intrinsics["fy"]
        self.cx = intrinsics.get("cx", intrinsics.get("ppx"))
        self.cy = intrinsics.get("cy", intrinsics.get("ppy"))

    def bbox_to_3d(self, bbox, depth_image):
        """
        bbox: [x1, y1, x2, y2] (normalized 0~1)
        depth_image: np.ndarray, 단위: meter

        return: (x, y, z) in camera frame
        """
        h, w = depth_image.shape
        x1, y1, x2, y2 = bbox
        cx_norm = (x1 + x2) / 2.0
        cy_norm = (y1 + y2) / 2.0

        u = int(cx_norm * w)
        v = int(cy_norm * h)

        z = float(depth_image[v, u])
        if z == 0.0:
            return None

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return (x, y, z)
