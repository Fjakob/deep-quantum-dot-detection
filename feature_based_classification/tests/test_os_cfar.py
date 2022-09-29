from unittest import TestCase
import numpy as np
from src.lib.peakDetectors.os_cfar import OS_CFAR


class TestCFAR(TestCase):
    """ OS-CFAR Test Suite """

    @classmethod
    def setUpClass(cls):
        global os_cfar
        os_cfar = OS_CFAR(N=4,T=1)

    ##########################################################################################
    #  T E S T    C A S E S
    ##########################################################################################

    def test_detect_one_peak(self):

        x = np.asarray([1., 7., 1., 1., 1.])
        detected_peak_indices, n_peak, _ = os_cfar.detect(x)

        self.assertEqual(n_peak, 1)
        self.assertListEqual(detected_peak_indices, [1])


    def test_detect_multiple_peaks(self):

        x = np.asarray([7., 1., 1., 1., 7.])
        detected_peak_indices, n_peak, _ = os_cfar.detect(x)

        self.assertEqual(n_peak, 2)
        self.assertListEqual(detected_peak_indices, [0, 4])

    
    def test_merge_peaks(self):

        x = np.asarray([1., 0., 8., 8., 9., 0., 0.])
        detected_peak_indices, n_peak, _ = os_cfar.detect(x)

        self.assertEqual(n_peak, 1)
        self.assertListEqual(detected_peak_indices, [4])

