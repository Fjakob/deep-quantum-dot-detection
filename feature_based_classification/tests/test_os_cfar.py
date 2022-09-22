import numpy as np
from src.lib.peakDetectors.os_cfar import OS_CFAR

def test_cfar_instanciation():
    detector = OS_CFAR()
    assert isinstance(detector, OS_CFAR)

def test_peak_detection():

    detector = OS_CFAR(N=4, T=1)

    # Test 1
    x1 = np.asarray([1., 7., 1., 1., 1.])
    detected_peak_indices, n_peak, _ = detector.detect(x1)

    assert n_peak == 1
    assert detected_peak_indices == [1]

    # Test 2
    x2 = np.asarray([7., 1., 1., 1., 7.])
    detected_peak_indices, n_peak, _ = detector.detect(x2)

    assert n_peak == 2
    assert detected_peak_indices == [0, 4]


    
