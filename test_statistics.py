"""Unit test for statistics.py functions."""
from unittest import TestCase
from statistics import average, variance, stdev
from math import sqrt


class StatisticsTest(TestCase):
    """Unit test of average, variance and stdev function."""

    def test_average_typical_values(self):
        """Test average with typical values."""
        self.assertEqual(average([1, 2, 3, 4, 5]), 3.0)
        self.assertEqual(average([10.0, 20.0, 30.0]), 20.0)
        self.assertEqual(average([1, 1, 1]), 1.0)

    def test_average_empty(self):
        """Test average with an empty list."""
        with self.assertRaises(ValueError):
            average([])

    def test_average_single_value(self):
        """Test average with a single value."""
        self.assertEqual(average([5]), 5.0)

    def test_average_large_numbers(self):
        """Test average with very large numbers."""
        self.assertEqual(average([1e6, 1e6 + 1, 1e6 - 1]), 1e6)

    def test_average_negative_numbers(self):
        """Test average with negative numbers."""
        self.assertEqual(average([-1, -2, -3, -4]), -2.5)

    def test_average_mixed_numbers(self):
        """Test average with mixed int and float numbers."""
        self.assertEqual(average([1, 2.5, 3, 4.75]), 2.8125)

    def test_variance_typical_values(self):
        """Variance of typical values."""
        self.assertEqual(0.0, variance([10.0, 10.0, 10.0, 10.0, 10.0]))
        self.assertEqual(2.0, variance([1, 2, 3, 4, 5]))
        self.assertEqual(8.0, variance([10, 2, 8, 4, 6]))

    def test_variance_decimal_values(self):
        """Test variance with decimal values."""
        self.assertAlmostEqual(variance([0.1, 4.1]), 4.0)
        self.assertAlmostEqual(variance([0.1, 4.1, 4.1, 8.1]), 8.0)

    def test_variance_single_element(self):
        """Test variance with a single element."""
        self.assertEqual(variance([5]), 0.0)

    def test_variance_non_integers(self):
        """Variance should work with decimal values."""
        # variance([x,y,z]) == variance([x+d,y+d,z+d]) for any d
        self.assertAlmostEqual(4.0, variance([0.1, 4.1]))
        # variance([0,4,4,8]) == 8
        self.assertAlmostEqual(8.0, variance([0.1, 4.1, 4.1, 8.1]))

    def test_variance_empty_list(self):
        """Variance of an empty list should raise an error."""
        with self.assertRaises(ValueError):
            variance([])

    def test_variance_single_value(self):
        """Variance of a single value should be zero."""
        self.assertEqual(0.0, variance([10.0]))

    def test_variance_large_values(self):
        """Variance of very large numbers."""
        self.assertAlmostEqual(2.6666666666666668e+16,
                               variance([1e8, 1e8 + 2e8, 1e8 - 2e8]))

    def test_variance_negative_numbers(self):
        """Test variance with negative numbers."""
        self.assertAlmostEqual(variance([-10, -2, -8, -4, -6]), 8.0)

    def test_variance_identical_values(self):
        """Test variance with identical values."""
        self.assertEqual(variance([42, 42, 42, 42]), 0.0)

    def test_variance_floats(self):
        """Test variance with floating-point numbers."""
        self.assertAlmostEqual(variance([1.5, 2.0, 2.5]),
                               0.16666666666666666)

    def test_variance_large_dataset(self):
        """Test variance with a large dataset."""
        large_data = [i for i in range(1000)]
        expected_variance = sum((x - 499.5) ** 2
                                for x in large_data) / len(large_data)
        self.assertAlmostEqual(variance(large_data),
                               expected_variance)

    def test_variance_integers_and_floats(self):
        """Test variance with a mix of integers and floats."""
        self.assertAlmostEqual(variance([1, 2.5, 3]),
                               0.7222222222222222)

    def test_stdev(self):
        """Standard deviation of typical values."""
        # standard deviation of a single value should be zero
        self.assertEqual(0.0, stdev([10.0]))
        # simple test
        self.assertEqual(2.0, stdev([1, 5]))
        # variance([0, 0.5, 1, 1.5, 2.0]) is 0.5
        self.assertEqual(sqrt(0.5), stdev([0, 0.5, 1, 1.5, 2]))

    def test_stdev_empty_list(self):
        """Standard deviation of an empty list should raise an error."""
        with self.assertRaises(ValueError):
            stdev([])

    def test_stdev_single_element(self):
        """Test standard deviation with a single element."""
        self.assertEqual(0.0, stdev([10]))

    def test_stdev_negative_numbers(self):
        """Test standard deviation with negative numbers."""
        self.assertAlmostEqual(2.8284271247461903,
                               stdev([-10, -2, -8, -4, -6]))

    def test_stdev_large_numbers(self):
        """Test standard deviation with very large numbers."""
        self.assertAlmostEqual(sqrt(2.6666666666666668e16),
                               stdev([1e8, 1e8 + 2e8, 1e8 - 2e8]))

    def test_stdev_identical_values(self):
        """Test standard deviation with identical values."""
        self.assertEqual(0.0, stdev([99, 99, 99, 99]))

    def test_stdev_floats(self):
        """Test standard deviation with floating-point numbers."""
        self.assertAlmostEqual(sqrt(variance([1.5, 2.0, 2.5])),
                               stdev([1.5, 2.0, 2.5]))


# if __name__ == '__main__':
#     import unittest
#
#     unittest.main(verbosity=1)
