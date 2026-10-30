"""Forehead two-finger trigger and Instant Transmission style teleport effect."""

from __future__ import annotations

import math

import cv2
import numpy as np


GOKU_SECONDS = 1.25
GOKU_HOLD_SECONDS = 1.0
FOREHEAD_LANDMARK = 151
FACE_EDGE_LANDMARKS = (234, 454)

_WHITE = np.array((255, 255, 255), dtype=np.float32)
_AURA = np.array((255, 222, 96), dtype=np.float32)  # BGR light blue


def _finger(points: np.ndarray, base: int) -> tuple[float, float, float]:
    """Return straightness, total length and MCP-to-tip reach of one finger."""
    segment = points[base:base + 4]
    length = float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())
    reach = float(np.linalg.norm(segment[3] - segment[0]))
    return (reach / length if length > 1e-6 else 0.0), length, reach


def forehead_salute(image_points: np.ndarray, world_points: np.ndarray,
                    forehead: np.ndarray, face_width: float) -> bool:
    """Index and middle straight and together, other fingers folded, tips on the forehead."""
    if image_points.shape != (21, 2) or world_points.shape != (21, 3) or face_width < 8.0:
        return False
    if not (np.isfinite(image_points).all() and np.isfinite(world_points).all()):
        return False
    index_straight, index_length, _ = _finger(world_points, 5)
    middle_straight, middle_length, _ = _finger(world_points, 9)
    if min(index_straight, middle_straight) < 0.80:
        return False
    reference = min(index_length, middle_length)
    if reference < 1e-6 or np.linalg.norm(world_points[8] - world_points[12]) > reference * 0.45:
        return False
    for base in (13, 17):
        straight, _, reach = _finger(world_points, base)
        if straight >= 0.80 and reach >= reference * 0.75:
            return False
    tips = (image_points[8] + image_points[12]) * 0.5
    return float(np.linalg.norm(tips - forehead)) < face_width * 0.6


def _smooth(edge0: float, edge1: float, value: float) -> float:
    value = min(1.0, max(0.0, (value - edge0) / (edge1 - edge0)))
    return value * value * (3.0 - 2.0 * value)


def teleport_envelopes(age: float, appearing: bool) -> tuple[float, float, float]:
    """Person opacity, speed-line strength and flash; appearing plays the timeline backwards."""
    t = min(1.0, max(0.0, age / GOKU_SECONDS))
    if appearing:
        t = 1.0 - t
    person = 1.0 - _smooth(0.42, 0.60, t)
    if 0.22 < t < 0.50:
        person *= 0.72 + 0.28 * math.sin(age * 70.0)
    lines = _smooth(0.0, 0.25, t) * (1.0 - _smooth(0.72, 1.0, t))
    flash = math.exp(-((t - 0.52) / 0.06) ** 2)
    return person, lines, flash


def render_teleport(frame: np.ndarray, background: np.ndarray, mask: np.ndarray | None,
                    age: float, appearing: bool) -> np.ndarray:
    person, lines, flash = teleport_envelopes(age, appearing)
    canvas = background.copy()
    if mask is None:
        return canvas
    points = cv2.findNonZero((mask > 0.02).astype(np.uint8))
    if points is None:
        return canvas
    # Work only around the silhouette: full-frame float filtering is too slow for live video.
    x, y, w, h = cv2.boundingRect(points)
    pad = max(24, w // 5)
    left, top = max(0, x - pad), max(0, y - pad)
    right, bottom = min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)
    area = np.s_[top:bottom, left:right]
    crop = mask[area]
    alpha = (crop * person)[:, :, None]
    output = frame[area].astype(np.float32) * alpha + background[area].astype(np.float32) * (1.0 - alpha)

    if lines > 0.01 or flash > 0.01:
        crop_h, crop_w = crop.shape
        small = cv2.resize(crop, (max(1, crop_w // 4), max(1, crop_h // 4)), interpolation=cv2.INTER_AREA)
        size = max(3, (w // 24) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        region = cv2.GaussianBlur(cv2.dilate(small, kernel), (0, 0), 1.5)
        region = cv2.resize(region, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        layer = np.zeros(crop.shape, dtype=np.float32)
        # Reseed ~30 times per second so the lines flicker like comic speed lines.
        rng = np.random.default_rng(int(age * 30))
        x0, y0 = x - left, y - top
        for _ in range(int(np.clip(w / 9, 12, 70))):
            line_x = int(rng.integers(x0 - w // 8, x0 + w + w // 8 + 1))
            line_top = int(rng.integers(y0 - h // 4, y0 + h))
            length = int(h * rng.uniform(0.25, 0.85))
            cv2.line(layer, (line_x, line_top), (line_x, line_top + length),
                     float(rng.uniform(0.55, 1.0)), int(rng.choice((1, 1, 2, 3))), cv2.LINE_AA)
        layer *= region
        glow = cv2.resize(layer, small.shape[::-1], interpolation=cv2.INTER_AREA)
        glow = cv2.resize(cv2.GaussianBlur(glow, (0, 0), 1.3), (crop_w, crop_h),
                          interpolation=cv2.INTER_LINEAR)
        output += (layer[:, :, None] * _WHITE + glow[:, :, None] * 1.6 * _AURA) * lines
        if flash > 0.01:
            halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.2), (crop_w, crop_h),
                              interpolation=cv2.INTER_LINEAR)
            output += halo[:, :, None] * _WHITE * flash * 0.85
    canvas[area] = np.clip(output, 0, 255).astype(np.uint8)
    return canvas
