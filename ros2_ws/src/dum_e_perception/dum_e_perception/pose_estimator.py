# pose_estimator.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import math

class PoseEstimator:
    def __init__(self, intrinsics: dict):
        self.fx = float(intrinsics["fx"])
        self.fy = float(intrinsics["fy"])
        self.cx = float(intrinsics.get("cx", intrinsics.get("ppx")))
        self.cy = float(intrinsics.get("cy", intrinsics.get("ppy")))

    def _pixel_to_camera(self, u: int, v: int, z: float) -> Tuple[float, float, float]:
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return (float(x), float(y), float(z))

    def bbox_to_3d(self, bbox, depth_image) -> Optional[Tuple[float, float, float]]:
        """기존: bbox 중앙 픽셀 1개"""
        h, w = depth_image.shape
        x1, y1, x2, y2 = bbox
        u = int(((x1 + x2) * 0.5) * w)
        v = int(((y1 + y2) * 0.5) * h)
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))

        z = float(depth_image[v, u])
        if z <= 0.0 or not np.isfinite(z):
            return None
        return self._pixel_to_camera(u, v, z)

    def bbox_to_3d_heuristic(
        self,
        bbox,
        depth_image: np.ndarray,
        *,
        roi_expand: float = 0.08,         # bbox를 조금 키워서(8%) 안정성 ↑
        z_min: float = 150.0,             # mm
        z_max: float = 2000.0,            # mm
        median_band: float = 30,          # median ± 3cm 범위 픽셀만 후보로
        min_valid_pixels: int = 30,       # ROI에 유효 픽셀이 너무 적으면 실패
    ) -> Optional[Tuple[float, float, float]]:
        """
        bbox: [x1,y1,x2,y2] normalized 0~1
        return: (x,y,z) camera frame
        """
        h, w = depth_image.shape
        x1n, y1n, x2n, y2n = bbox

        # 1) ROI를 약간 확장
        bw = x2n - x1n
        bh = y2n - y1n
        cx = (x1n + x2n) * 0.5
        cy = (y1n + y2n) * 0.5
        ex = bw * roi_expand
        ey = bh * roi_expand

        x1n2 = max(0.0, cx - bw * 0.5 - ex)
        x2n2 = min(1.0, cx + bw * 0.5 + ex)
        y1n2 = max(0.0, cy - bh * 0.5 - ey)
        y2n2 = min(1.0, cy + bh * 0.5 + ey)

        x1 = int(round(x1n2 * w))
        x2 = int(round(x2n2 * w))
        y1 = int(round(y1n2 * h))
        y2 = int(round(y2n2 * h))

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        roi = depth_image[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # 2) 유효 depth 마스크
        valid = np.isfinite(roi) & (roi > z_min) & (roi < z_max)
        if int(valid.sum()) < min_valid_pixels:
            # fallback: 중앙 근처에서 비어있지 않은 depth를 찾는 간단 탐색
            return self._fallback_center_search(bbox, depth_image, z_min=z_min, z_max=z_max)

        valid_vals = roi[valid]
        z_med = float(np.median(valid_vals))

        # 3) median 근처 band 안의 픽셀들 중 ROI 중심에 가장 가까운 픽셀 선택
        band = valid & (np.abs(roi - z_med) <= median_band)
        if int(band.sum()) < max(10, min_valid_pixels // 3):
            # band가 너무 적으면 valid 전체에서 선택
            band = valid

        # ROI 좌표계에서 중심
        rc = (roi.shape[0] - 1) * 0.5
        cc = (roi.shape[1] - 1) * 0.5

        ys, xs = np.where(band)
        if len(xs) == 0:
            return self._fallback_center_search(bbox, depth_image, z_min=z_min, z_max=z_max)

        # 중심과의 거리 최소
        d2 = (ys - rc) ** 2 + (xs - cc) ** 2
        idx = int(np.argmin(d2))
        u = x1 + int(xs[idx])
        v = y1 + int(ys[idx])

        z = float(depth_image[v, u])
        if z <= 0.0 or not np.isfinite(z):
            return self._fallback_center_search(bbox, depth_image, z_min=z_min, z_max=z_max)

        return self._pixel_to_camera(u, v, z)

    def _fallback_center_search(
        self,
        bbox,
        depth_image: np.ndarray,
        *,
        z_min: float,
        z_max: float,
        max_radius_px: int = 40,
    ) -> Optional[Tuple[float, float, float]]:
        """bbox 중심에서 주변을 원형으로 조금씩 탐색하며 depth 유효 픽셀 찾기"""
        h, w = depth_image.shape
        x1, y1, x2, y2 = bbox
        uc = int(((x1 + x2) * 0.5) * w)
        vc = int(((y1 + y2) * 0.5) * h)
        uc = max(0, min(w - 1, uc))
        vc = max(0, min(h - 1, vc))

        for r in range(0, max_radius_px + 1, 2):
            # 사각 링을 샘플링(빠르고 간단)
            for du in range(-r, r + 1, 2):
                for dv in (-r, r):
                    u = max(0, min(w - 1, uc + du))
                    v = max(0, min(h - 1, vc + dv))
                    z = float(depth_image[v, u])
                    if np.isfinite(z) and (z_min < z < z_max):
                        return self._pixel_to_camera(u, v, z)

            for dv in range(-r, r + 1, 2):
                for du in (-r, r):
                    u = max(0, min(w - 1, uc + du))
                    v = max(0, min(h - 1, vc + dv))
                    z = float(depth_image[v, u])
                    if np.isfinite(z) and (z_min < z < z_max):
                        return self._pixel_to_camera(u, v, z)

        return None
