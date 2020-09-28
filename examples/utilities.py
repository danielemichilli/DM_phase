import numpy as np
import matplotlib.pyplot as plt

from DM_phase import dedisperse_waterfall


def simulate_burst(
    components=2, 
    nchan=1024, 
    nbin=2000, 
    freq_range=[1200., 1600.], 
    t_res=1e-5,
    DM = 125.
):
    """Simulate a burst with multiple components.

    Parameters
    ----------
    components : int
        Number of components in the burst.
    nchan : int
        Number of frequency channels.
    nbin : int
        Number of time bins.
    freq_range : list
        Frequency of the two edge channels, in MHz.
    t_res : float
        Time resolution of one time bin, in seconds.
    DM : float
        Dispersion measure of the burst, in pc/cc/

    Returns
    -------
    freq : ndarray
        Array of frequency values of each channel
    power : ndarray
        Waterfall of the burst in S/N.

    """

    def gaus(x, a, m, s):
        return np.sqrt(a)*np.exp(-(x-m)**2/(2*s**2))

    xx, yy = np.meshgrid(np.arange(nbin), np.arange(nchan))
    
    power = np.random.normal(size=xx.shape)
    width = nbin // 50
    band = nchan // (2 * components)
    peak_snr = 10
    for i,n in enumerate(np.arange(-(components-1)/2, +components/2)):
        loc_t = nbin // 2 + n * width * 3
        loc_f = nchan // 2 + n * nchan / (components + 1)
        peak = peak_snr * (1 - i % 2 * 0.5)
        power += gaus(xx, peak, loc_t, width) * gaus(yy, peak, loc_f, band)
    
    freq = np.linspace(freq_range[0], freq_range[1], nchan)
    power = dedisperse_waterfall(power, -DM, freq, t_res)
    return freq, power
    

def downsample_waterfall(wfall, t_scrunch=2, f_scrunch=16):
    # Downsample a waterfall
    wfall = wfall[: wfall.shape[0] // f_scrunch * f_scrunch, : wfall.shape[1] // t_scrunch * t_scrunch]
    wfall = wfall.reshape([wfall.shape[0] // f_scrunch, f_scrunch, wfall.shape[1]]).mean(axis=1)
    wfall = wfall.reshape([wfall.shape[0], wfall.shape[1] // t_scrunch, t_scrunch]).mean(axis=-1)
    return wfall


