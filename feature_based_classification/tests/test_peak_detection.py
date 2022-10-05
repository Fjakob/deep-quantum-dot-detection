from unittest import TestCase
import numpy as np
from src.lib.peakDetectors.os_cfar import OS_CFAR
from src.lib.peakDetectors.threshold_detector import ThresholdDetector


class TestThresholdDetector(TestCase):
    """ Threshold Detector Test Suite """

    @classmethod
    def setUpClass(cls):
        global threshold_detector
        threshold_detector = ThresholdDetector(threshold=1)

    ##########################################################################################
    #  T E S T    C A S E S
    ##########################################################################################

    def test_threshold(self):

        x = np.zeros((10, 1))
        _, _, threshold = threshold_detector.detect(x)

        self.assertEqual(len(threshold), 10)
        self.assertTrue((threshold == threshold_detector.threshold).all())


    def test_detect_one_peak(self):

        x = np.asarray([.5, .6, .4, 1.3, .9])
        detected_peak_indices, n_peak, _ = threshold_detector.detect(x)

        self.assertEqual(n_peak, 1)
        self.assertListEqual(detected_peak_indices, [3])


    def test_detect_multiple_peaks(self):

        x = np.asarray([1.1, .6, .4, 1.3, 1.2])
        detected_peak_indices, n_peak, _ = threshold_detector.detect(x)

        self.assertEqual(n_peak, 2)
        self.assertListEqual(detected_peak_indices, [0, 3])


    def test_optimization(self):

        X = np.asarray([[.5, .5, .5, 1., .5], 
                        [1., .9, 1., .9, .9]])
        Y = np.asarray([1,2])
        threshold_detector.optimize_parameters((X,Y))

        self.assertAlmostEqual(threshold_detector.threshold, 0.95, delta=0.05)



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

