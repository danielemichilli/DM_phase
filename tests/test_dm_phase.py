"""Unit and science tests for dm_phase.py."""

import numpy as np
import os
import pytest

from DM_phase import get_dm


@pytest.fixture
def waterfall():
    """Bursts from B0355+54 detected by CHIME/FRB."""
    waterfall = np.load("B0355+54_DM57.2403.npy")

    # calculate per-channel mean and variance
    mean = np.nanmean(waterfall, axis=1)
    var = np.nanvar(waterfall, axis=1)

    # subtract per-channel mean
    waterfall -= mean[:, None]
    # divide by per-channel standard deviation
    waterfall /= np.sqrt(var[:, None])

    median = np.nanmedian(waterfall)
    waterfall[np.isnan(waterfall)] = median

    return waterfall


def test_get_dm(waterfall):
    """Test phase-DM determination."""

    nchan = waterfall.shape[0]

    # source DM
    source_dm = 57.1420

    # data
    data_dm = 57.2403
    ddm = 5.0

    # CHIME/FRB
    sampling_time = 0.00098304  # s
    fbottom = 400.1953125  # MHz
    ftop = 800.1953125  # MHz
    df = (ftop - fbottom) / nchan

    center_frequencies = np.arange(fbottom + df / 2.0, ftop, df)

    trials = 100

    dms = np.linspace(data_dm - ddm, data_dm + ddm, trials)

    dm, dm_e = get_dm(
        waterfall,
        dms - data_dm,
        sampling_time,
        center_frequencies,
        fname="B0355+54_test",
        manual_cutoff=False,
        manual_bandwidth=False,
        ref_freq="center",
    )

    dm += data_dm

    print("Phase structure-optimizing DM is {:.2f} +/- {:.2f}..".format(dm, dm_e))

    # check if DM and uncertainty are OK
    assert abs(source_dm - dm) < dm_e
    assert dm_e < 0.1

    # check if diagnostic plots are created
    assert os.path.isfile("B0355+54_test_DM_Search.pdf")
    assert os.path.isfile("B0355+54_test_Waterfall_5sig.pdf")

    # clean up
    os.remove("B0355+54_test_DM_Search.pdf")
    os.remove("B0355+54_test_Waterfall_5sig.pdf")
