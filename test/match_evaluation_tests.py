import unittest
import numpy as np
from scipy.stats import norm

from tools.match_evaluation import get_player_final_utilities_from_log_file, get_player_utilities_from_log_file, get_logs_data, calculate_confidence_interval


class MatchEvaluationTests(unittest.TestCase):
    def test_log_reading(self):
        log_file_path = 'test/sample_log-large.log'
        final_scores, final_scores_player_names = get_player_final_utilities_from_log_file(log_file_path)
        utilities, utilities_player_names = get_player_utilities_from_log_file(log_file_path)

        self.assertEqual(utilities.shape, (50000, 2))
        self.assertEqual(final_scores_player_names, utilities_player_names)
        self.assertTrue(np.all(final_scores == np.sum(utilities, axis=0)))

    def test_confidence_interval_calculation(self):
        log_files_paths = [
            'test/match_evaluation/sample_log_1.log',
            'test/match_evaluation/sample_log_2.log'
        ]
        log_readings = [get_player_utilities_from_log_file(log_file_path) for log_file_path in log_files_paths]
        data = get_logs_data(*log_readings)
        means, interval_half_size, lower_bounds, upper_bounds = calculate_confidence_interval(data, 0.95)
        self.assertEqual(means.tolist(), [1, -1])

        sigma = np.sqrt(8)
        sme = sigma / np.sqrt(10)
        z = norm.ppf(0.975)

        interval_size = z * sme

        self.assertEqual(interval_half_size.tolist(), [interval_size] * 2)
        self.assertEqual(lower_bounds.tolist(), [mean - interval_size for mean in means])
        self.assertEqual(upper_bounds.tolist(), [mean + interval_size for mean in means])
