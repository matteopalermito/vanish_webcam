import unittest
from unittest.mock import patch

import numpy as np

import vanish_cam as app
from goku_effect import GOKU_HOLD_SECONDS, GOKU_SECONDS, forehead_salute, render_teleport


def salute_hand(spread=0.0, ring_extended=False):
    points = np.zeros((21, 3))
    straight = ((0.0, 1.0, 0.0), (0.0, 1.7, 0.0), (0.0, 2.15, 0.0), (0.0, 2.5, 0.0))
    folded = ((0.0, 1.0, 0.0), (0.0, 1.5, 0.2), (0.0, 1.3, 0.5), (0.0, 0.95, 0.3))
    for base, x, extended, finger_spread in (
        (5, -0.1, True, -spread), (9, 0.1, True, spread),
        (13, 0.3, ring_extended, 0.0), (17, 0.5, False, 0.0),
    ):
        for offset, (px, py, pz) in enumerate(straight if extended else folded):
            lean = finger_spread * (py - 1.0) if extended else 0.0
            points[base + offset] = (x + px + lean, py, pz)
    return points


def image_points(world):
    # Image y grows downward, so flip the world y axis.
    return world[:, :2] * (40, -40) + (300, 300)


class ForeheadSaluteTests(unittest.TestCase):
    def setUp(self):
        self.world = salute_hand()
        self.image = image_points(self.world)
        self.tips = (self.image[8] + self.image[12]) / 2

    def test_fingers_together_on_forehead_trigger(self):
        self.assertTrue(forehead_salute(self.image, self.world, self.tips + (0, 5), 120.0))

    def test_pose_away_from_forehead_does_not_trigger(self):
        self.assertFalse(forehead_salute(self.image, self.world, self.tips + (0, 200), 120.0))

    def test_v_sign_or_open_hand_does_not_trigger(self):
        for world in (salute_hand(spread=0.6), salute_hand(ring_extended=True)):
            with self.subTest():
                image = image_points(world)
                tips = (image[8] + image[12]) / 2
                self.assertFalse(forehead_salute(image, world, tips, 120.0))

    def test_invalid_input_does_not_trigger(self):
        self.assertFalse(forehead_salute(self.image[:20], self.world, self.tips, 120.0))
        self.assertFalse(forehead_salute(self.image, np.full((21, 3), np.nan), self.tips, 120.0))
        self.assertFalse(forehead_salute(self.image, self.world, self.tips, 0.0))


class GokuHoldTests(unittest.TestCase):
    def hold(self, state, posed, now):
        with patch("builtins.print"):
            return app.update_goku_hold(state, posed, now)

    def test_triggers_once_after_hold_then_requires_release(self):
        state = app.AppState(phase="ready")
        self.assertFalse(self.hold(state, True, 0.0))
        self.assertFalse(self.hold(state, True, GOKU_HOLD_SECONDS * 0.5))
        self.assertTrue(self.hold(state, True, GOKU_HOLD_SECONDS + 0.01))
        self.assertFalse(self.hold(state, True, GOKU_HOLD_SECONDS + 2.0))
        self.assertFalse(self.hold(state, False, GOKU_HOLD_SECONDS + 3.0))
        self.assertFalse(self.hold(state, True, 10.0))
        self.assertTrue(self.hold(state, True, 10.0 + GOKU_HOLD_SECONDS))

    def test_short_dropout_is_tolerated_but_long_break_restarts(self):
        state = app.AppState(phase="ready")
        self.hold(state, True, 0.0)
        self.hold(state, True, 0.3)
        self.hold(state, False, 0.45)
        self.assertTrue(self.hold(state, True, GOKU_HOLD_SECONDS))
        state = app.AppState(phase="ready")
        self.hold(state, True, 0.0)
        self.hold(state, False, 0.5)
        self.assertFalse(self.hold(state, True, GOKU_HOLD_SECONDS))


class TeleportRenderTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((120, 160, 3), 200, np.uint8)
        self.background = np.full((120, 160, 3), 30, np.uint8)
        self.mask = np.zeros((120, 160), np.float32)
        self.mask[20:110, 60:100] = 1.0

    def test_vanish_ends_on_background_and_appear_ends_on_person(self):
        gone = render_teleport(self.frame, self.background, self.mask, GOKU_SECONDS, False)
        np.testing.assert_array_equal(gone, self.background)
        back = render_teleport(self.frame, self.background, self.mask, GOKU_SECONDS, True)
        np.testing.assert_array_equal(back[30:100, 65:95], self.frame[30:100, 65:95])
        np.testing.assert_array_equal(back[:, :20], self.background[:, :20])

    def test_mid_effect_draws_speed_lines_and_handles_missing_mask(self):
        middle = render_teleport(self.frame, self.background, self.mask, GOKU_SECONDS * 0.4, False)
        self.assertGreater(int(middle[20:110, 60:100].max()), 200)
        empty = render_teleport(self.frame, self.background, None, 0.5, False)
        np.testing.assert_array_equal(empty, self.background)


if __name__ == "__main__":
    unittest.main()
