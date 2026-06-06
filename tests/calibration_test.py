import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from nf_robot.host import calibration
from nf_robot.host import eyelet_calibration

class TestCalibration(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset for testing
        # 4 anchors, 2 markers ('origin' and 'tag1')
        self.averages = {
            'origin': np.zeros((4, 1, 2, 3)), # for each of the four anchors, a list of one pose at 0,0,0 with no rotation.
            'tag1': [
                [(np.zeros(3), np.array([1.0, 0.0, 0.0]))], # Seen by Anchor 0
                [(np.zeros(3), np.array([2.0, 0.0, 0.0]))], # Seen by Anchor 1
                [],
                []
            ]
        }
        
        # Helper to create a flat x vector representing 4 anchors
        # Shape: (4, 2, 3) -> flattened
        # Each anchor is ((rvec), (tvec))
        self.anchors_zero = np.zeros((4, 2, 3))
        self.x_zero = self.anchors_zero.flatten()

    @patch('nf_robot.host.calibration.compose_poses')
    @patch('nf_robot.host.calibration.model_constants')
    def test_multi_card_residuals_origin_constraint(self, mock_constants, mock_compose):
        """
        Test that origin markers produce weighted residuals based on distance from [0,0,0].
        """
        # Setup Mocks
        mock_constants.anchor_camera = np.zeros((2, 3))
        
        # Define compose_poses behavior: return the translation part of the marker pose directly
        # to simulate the anchor being at identity.
        # compose_poses signature is list of poses -> returns pose (r, t)
        def side_effect_compose(poses):
            # poses[0] is anchor, poses[1] is const, poses[2] is marker
            # Return marker pose to simulate identity anchor
            return poses[2] 
        
        mock_compose.side_effect = side_effect_compose

        # Execute
        residuals = calibration.multi_card_residuals(self.x_zero, {'origin': self.averages['origin']})

        # Analysis
        # The origin marker is at [0,0,0]. Residual should be 0.
        # But wait, logic is (pos - 0) * 5.0. 
        # If pos is 0, residual is 0.
        
        # Let's change the input to have an error
        # Origin marker seen at [1, 2, 3]
        bad_origin_data = {
            'origin': [[(np.zeros(3), np.array([1.0, 2.0, 3.0]))], [], [], []]
        }
        
        residuals = calibration.multi_card_residuals(self.x_zero, bad_origin_data)
        
        # Expected: ([1, 2, 3] - [0, 0, 0]) * W_ORIGIN
        expected_residuals = np.array([1, 2, 3])*calibration.W_ORIGIN
        
        # Note: residuals array will also contain Z-constraints for the anchors (0 deviation)
        # Slicing the first 3 elements which correspond to the marker
        np.testing.assert_array_almost_equal(residuals[:3], expected_residuals)

    @patch('nf_robot.host.calibration.compose_poses')
    @patch('nf_robot.host.calibration.model_constants')
    def test_multi_card_residuals_consistency_constraint(self, mock_constants, mock_compose):
        """
        Test that shared markers minimize distance to their centroid.
        """
        mock_constants.anchor_camera = np.zeros((2, 3))
        
        # Scenario: 
        # Anchor 0 sees tag1 at [10, 0, 0]
        # Anchor 1 sees tag1 at [12, 0, 0]
        # Centroid is [11, 0, 0]
        # Errors: [10-11, 0, 0] and [12-11, 0, 0] -> [-1, 0, 0] and [1, 0, 0]
        
        data = {
            'tag1': [
                [(np.zeros(3), np.array([10.0, 0.0, 0.0]))], 
                [(np.zeros(3), np.array([12.0, 0.0, 0.0]))],
                [], 
                []
            ]
        }

        def side_effect_compose(poses):
            return poses[2] # Pass through marker pose
        mock_compose.side_effect = side_effect_compose

        residuals = calibration.multi_card_residuals(self.x_zero, data)

        # There are 2 sightings * 3 coords = 6 residuals for the marker
        marker_residuals = residuals[:6]
        
        expected = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(marker_residuals, expected)

    @patch('nf_robot.host.calibration.compose_poses')
    @patch('nf_robot.host.calibration.model_constants')
    def test_planarity_residual_is_not_repeated_per_marker(self, mock_constants, mock_compose):
        """
        Extra marker entries should not multiply the anchor z-plane constraint.
        """
        mock_constants.anchor_camera = np.zeros((2, 3))
        mock_compose.side_effect = lambda poses: poses[2]

        anchors = np.zeros((4, 2, 3))
        anchors[:, 1, 2] = np.array([0.0, 1.0, 2.0, 3.0])
        x = anchors.flatten()

        one_marker = {
            'origin': [[(np.zeros(3), np.zeros(3))], [], [], []],
        }
        three_markers = {
            'origin': [[(np.zeros(3), np.zeros(3))], [], [], []],
            'tag1': [[(np.zeros(3), np.array([1.0, 0.0, 0.0]))], [], [], []],
            'tag2': [[(np.zeros(3), np.array([2.0, 0.0, 0.0]))], [], [], []],
        }

        residuals_one = calibration.multi_card_residuals(x, one_marker)
        residuals_three = calibration.multi_card_residuals(x, three_markers)
        expected_plane = (anchors[:, 1, 2] - np.mean(anchors[:, 1, 2])) * calibration.W_PLANAR

        self.assertEqual(len(residuals_one), 7)
        self.assertEqual(len(residuals_three), 7)
        np.testing.assert_array_almost_equal(residuals_one[-4:], expected_plane)
        np.testing.assert_array_almost_equal(residuals_three[-4:], expected_plane)

    @patch('nf_robot.host.calibration.optimize.least_squares')
    @patch('nf_robot.host.calibration.invert_pose')
    @patch('nf_robot.host.calibration.compose_poses')
    @patch('nf_robot.host.calibration.model_constants')
    def test_optimize_anchor_poses_success(self, mock_constants, mock_compose, mock_invert, mock_least_squares):
        """
        Test the optimization driver function calls the solver and returns reshaped poses.
        """
        # Setup mocks
        mock_constants.anchor_camera = np.zeros((2, 3))
        mock_compose.return_value = "composed_pose"
        mock_invert.return_value = np.zeros((2, 3)) # Initial guess
        
        # Mock successful solver result
        mock_result = MagicMock()
        mock_result.success = True
        # Result x should be flattened 4 anchors * 6 params
        mock_result.x = np.ones(24)
        mock_result.fun = np.zeros(24)
        mock_result.cost = 0.0
        mock_result.status = 1
        mock_result.nfev = 3
        mock_result.optimality = 0.0
        mock_least_squares.return_value = mock_result

        # Execute
        result_poses = calibration.optimize_anchor_poses(self.averages)

        # Verify initial guess construction
        # Should be called 4 times (once per anchor)
        self.assertEqual(mock_invert.call_count, 4)
        
        # Verify solver call
        mock_least_squares.assert_called_once()
        args, kwargs = mock_least_squares.call_args
        self.assertEqual(kwargs['method'], 'trf')
        self.assertEqual(kwargs['loss'], calibration.ROBUST_LOSS)
        self.assertEqual(kwargs['f_scale'], calibration.ROBUST_F_SCALE)
        
        # Verify return shape (4, 2, 3)
        self.assertEqual(result_poses.shape, (4, 2, 3))
        self.assertTrue(np.all(result_poses == 1)) # Based on our mock result.x

    @patch('nf_robot.host.calibration.optimize.least_squares')
    @patch('nf_robot.host.calibration.invert_pose')
    @patch('nf_robot.host.calibration.compose_poses')
    @patch('nf_robot.host.calibration.model_constants')
    def test_optimize_anchor_poses_failure(self, mock_constants, mock_compose, mock_invert, mock_least_squares):
        """
        Test that optimization failure returns None.
        """
        mock_invert.return_value = np.zeros((2, 3))
        
        # Mock failed solver result
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.status = -1
        mock_result.message = "Failed to converge"
        mock_result.fun = np.zeros(24)
        mock_result.cost = 0.0
        mock_result.nfev = 3
        mock_result.optimality = 0.0
        mock_least_squares.return_value = mock_result

        # Execute
        result = calibration.optimize_anchor_poses(self.averages)

        # Verify
        self.assertIsNone(result)


class TestEyeletCalibration(unittest.TestCase):
    @patch('nf_robot.host.eyelet_calibration.compose_poses')
    @patch('nf_robot.host.eyelet_calibration.model_constants')
    def test_planarity_residual_is_not_repeated_per_marker(self, mock_constants, mock_compose):
        """
        Arpeggio anchor/eyelet z-plane residuals should be added once per solve,
        not once per observed marker.
        """
        mock_constants.arp_anchor_camera = np.zeros((2, 3))
        mock_compose.side_effect = lambda poses: poses[-1]

        fixed_anchor_poses = np.zeros((2, 2, 3))
        fixed_anchor_poses[:, 1, 2] = np.array([0.0, 2.0])
        eyelets = np.array([[0.0, 0.0, 4.0], [0.0, 1.0, 6.0]])
        x = eyelets.flatten()

        one_marker = {
            'origin': [[(np.zeros(3), np.zeros(3))], []],
        }
        three_markers = {
            'origin': [[(np.zeros(3), np.zeros(3))], []],
            'tag1': [[(np.zeros(3), np.array([1.0, 0.0, 0.0]))], []],
            'tag2': [[(np.zeros(3), np.array([2.0, 0.0, 0.0]))], []],
        }

        residuals_one = eyelet_calibration.multi_card_residuals(
            x,
            one_marker,
            diamond_observations=None,
            fixed_anchor_poses=fixed_anchor_poses,
        )
        residuals_three = eyelet_calibration.multi_card_residuals(
            x,
            three_markers,
            diamond_observations=None,
            fixed_anchor_poses=fixed_anchor_poses,
        )

        all_zs = np.array([0.0, 2.0, 4.0, 6.0])
        expected_plane = (all_zs - np.mean(all_zs)) * eyelet_calibration.W_PLANAR

        self.assertEqual(len(residuals_one), 8)
        self.assertEqual(len(residuals_three), 8)
        np.testing.assert_array_almost_equal(residuals_one[3:7], expected_plane)
        np.testing.assert_array_almost_equal(residuals_three[3:7], expected_plane)

    @patch('nf_robot.host.eyelet_calibration.compose_poses')
    @patch('nf_robot.host.eyelet_calibration.model_constants')
    def test_eyelet_line_geometry_adds_only_usable_line_residuals(self, mock_constants, mock_compose):
        fixed_anchor_poses = np.zeros((2, 2, 3))
        eyelets = np.array([[0.0, 0.0, 4.0], [0.0, 1.0, 6.0]])
        x = eyelets.flatten()
        mock_constants.arp_anchor_right_eyelet = (np.zeros(3), np.array([1.0, 0.0, 0.0]))
        mock_compose.side_effect = lambda poses: poses[-1]

        baseline = eyelet_calibration.multi_card_residuals(
            x,
            {},
            diamond_observations=None,
            fixed_anchor_poses=fixed_anchor_poses,
        )
        with_line_geometry = eyelet_calibration.multi_card_residuals(
            x,
            {},
            diamond_observations=None,
            fixed_anchor_poses=fixed_anchor_poses,
            line_geometry={
                "gantry_pos": [0.0, 0.0, 0.0],
                "line_lengths": [0.5, 4.0, 1.0, 1.0],
                "usable_lines": [True, True, False, False],
            },
        )

        self.assertEqual(len(with_line_geometry), len(baseline) + 2)
        self.assertNotEqual(float(with_line_geometry[4]), 0.0)
        self.assertEqual(float(with_line_geometry[5]), 0.0)

    @patch('nf_robot.host.eyelet_calibration.optimize.least_squares')
    def test_optimize_arp_anchors_uses_robust_solver(self, mock_least_squares):
        fixed_anchor_poses = np.zeros((2, 2, 3))
        initial_eyelets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.x = initial_eyelets.flatten()
        mock_result.fun = np.zeros(8)
        mock_result.cost = 0.0
        mock_result.status = 1
        mock_result.nfev = 2
        mock_result.optimality = 0.0
        mock_least_squares.return_value = mock_result

        anchors, eyelets = eyelet_calibration.optimize_arp_anchors(
            raw_obs={},
            diamond_observations=None,
            initial_eyelet_guesses=initial_eyelets,
            fixed_anchor_poses=fixed_anchor_poses,
        )

        self.assertIs(anchors, fixed_anchor_poses)
        np.testing.assert_array_equal(eyelets, initial_eyelets)
        args, kwargs = mock_least_squares.call_args
        self.assertEqual(kwargs['method'], 'trf')
        self.assertEqual(kwargs['loss'], eyelet_calibration.ROBUST_LOSS)
        self.assertEqual(kwargs['f_scale'], eyelet_calibration.ROBUST_F_SCALE)
