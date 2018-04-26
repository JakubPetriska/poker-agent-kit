import unittest
from unittest import TestSuite

from test.hand_evaluation_tests import HandEvaluationTests
from test.exploitability_tests import ExploitabilityTests
from test.cfr_tests import CfrTests
from test.best_response_player_utility_tests import BestResponsePlayerUtilityTests
from test.utils_tests import UtilsTests
from test.sampling_tests import SamplingTests
from test.data_biased_response_tests import DataBiasedResponseTests

test_classes = [
    HandEvaluationTests,
    ExploitabilityTests,
    BestResponsePlayerUtilityTests,
    UtilsTests,
    SamplingTests,
    CfrTests,
    DataBiasedResponseTests,
]


def load_tests(loader, tests, pattern):
    suite = TestSuite()
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
