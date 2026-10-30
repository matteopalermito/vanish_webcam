import unittest
from unittest.mock import Mock, patch

import numpy as np

from goku_effect import GOKU_HOLD_SECONDS, GOKU_SECONDS, forehead_salute, render_teleport
from modes import GokuMode
from test_video_output import run_mock_loop


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
    def hold(self, mode, posed, now):
        with patch("builtins.print"):
            return mode.update_hold(posed, now)

    def test_triggers_once_after_hold_then_requires_release(self):
        mode = GokuMode(Mock())
        self.assertFalse(self.hold(mode, True, 0.0))
        self.assertFalse(self.hold(mode, True, GOKU_HOLD_SECONDS * 0.5))
        self.assertTrue(self.hold(mode, True, GOKU_HOLD_SECONDS + 0.01))
        self.assertFalse(self.hold(mode, True, GOKU_HOLD_SECONDS + 2.0))
        self.assertFalse(self.hold(mode, False, GOKU_HOLD_SECONDS + 3.0))
        self.assertFalse(self.hold(mode, True, 10.0))
        self.assertTrue(self.hold(mode, True, 10.0 + GOKU_HOLD_SECONDS))

    def test_short_dropout_is_tolerated_but_long_break_restarts(self):
        mode = GokuMode(Mock())
        self.hold(mode, True, 0.0)
        self.hold(mode, True, 0.3)
        self.hold(mode, False, 0.45)
        self.assertTrue(self.hold(mode, True, GOKU_HOLD_SECONDS))
        mode = GokuMode(Mock())
        self.hold(mode, True, 0.0)
        self.hold(mode, False, 0.5)
        self.assertFalse(self.hold(mode, True, GOKU_HOLD_SECONDS))


class GokuVideoLoopTests(unittest.TestCase):
    def test_gesture_vanishes_and_brings_back_while_v_and_r_are_ignored(self):
        frames = [np.full((24, 40, 3), 20 * (i + 1), np.uint8) for i in range(9)]
        times = (0.1, 4.1, 4.3, 5.35, 6.7, 6.8, 7.0, 8.05, 9.4)
        keys = (0, 0, 0, 0, 0, ord("r"), ord("v"), 0, ord("q"))
        detector = Mock()
        # Hold, fire, release while gone, hold again, fire to reappear.
        detector.detect.side_effect = [True, True, False, True, True]
        displayed = run_mock_loop(self, GokuMode(detector), frames, times, keys,
                                  Mock(width=40, height=24))
        background = frames[1]
        np.testing.assert_array_equal(displayed[2], frames[2])
        np.testing.assert_array_equal(displayed[4], background)
        np.testing.assert_array_equal(displayed[5], background)
        np.testing.assert_array_equal(displayed[6], background)  # R did not bring the person back.
        np.testing.assert_array_equal(displayed[8], frames[8])  # Reappeared by gesture.
        self.assertEqual(detector.detect.call_count, 5)
        detector.close.assert_called_once()


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
