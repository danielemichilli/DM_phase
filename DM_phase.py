"""
Incoherent search for best Dispersion Measure from a PSRCHIVE file.
The search uses phase information and thus it is not sensitive to Radio Frequency Interference or complex spectro-temporal pulse shape.
"""

import os
import argparse

import psrchive
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import scipy.signal
from scipy.fftpack import fft, ifft


plt.rcParams['toolbar'] = 'None'

def _load_psrchive(fname):
    """
    Load data from a PSRCHIVE file.
    """
    
    archive = psrchive.Archive_load(fname)
    archive.set_dedispersed(False)  # Un-dedisperse
    archive.update()
    archive.pscrunch()
    archive.tscrunch()
    archive.remove_baseline()
    w = archive.get_weights().squeeze()
    waterfall = archive.get_data().squeeze()
    waterfall *= w[:, np.newaxis]
    f_ch = [archive.get_first_Integration().get_centre_frequency(i) for i in range(archive.get_nchan())]
    dt = archive.get_time_resolution()
    
    if archive.get_bandwidth() < 0: 
        waterfall = np.flipud(waterfall)
        f_ch = f_ch[::-1]

    return waterfall, f_ch, dt

def _get_Spect(waterfall):
    """
    Get the coherent spectrum of the waterfall.
    """
    
    FT = fft(waterfall)
    amp = np.abs(FT)
    amp[amp == 0] = 1
    spect = np.sum(FT / amp, axis=0)
    return spect

def _get_Pow(waterfall):
    """
    Get the coherent power of the waterfall.
    """
    
    spect = _get_Spect(waterfall)
    Pow = np.abs(spect)**2
    return Pow

def _get_f_threshold(Pow_list, MEAN, STD):
    """
    Get the Fourier frequency cutoff.
    """

    s   = np.max(Pow_list, axis=1)
    SN  = (s - MEAN) / STD 
    Kern = np.round(_get_Window(SN * (SN > 5)) / 2.).astype(int)
    if Kern < 5: Kern = 5
    return Kern

def _get_f_threshold_manual(Pow_list, dPow_list):


    fig = plt.figure(figsize=(6, 8.5), facecolor='k')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, hspace=0)
    gs = gridspec.GridSpec(2, 1, hspace=0, height_ratios=[1, 4])
    ax_prof = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1], sharex=ax_prof)

    fig.suptitle("Left click to zoom in, right click to zoom out.\nClose when satisfied.", color='w')
    ax_prof.axis('off')
    ax_map.axis('off')

    im = ax_map.imshow(Pow_list, origin='lower', aspect='auto', cmap='YlOrBr_r', interpolation='spline16')
    ax_map.set_ylim([2, Pow_list.shape[0]])
    prof = dPow_list.sum(axis=0)
    plot, = ax_prof.plot(prof, 'w-', linewidth=2, clip_on=False)
    ax_prof.set_ylim([prof.min(), prof.max()])

    lim_list = [Pow_list.shape[0],]
    def onclick(event):
        if event.button == 1:
            y = int(round(event.ydata))
            lim_list.append(y)
            ax_map.set_ylim([2, y])
            prof = dPow_list[:y].sum(axis=0)
            plot.set_ydata(prof)
            ax_prof.set_ylim([prof.min(), prof.max()])
        elif (event.button == 2) or (event.button == 3):
            if len(lim_list) > 1: del lim_list[-1]
            y = lim_list[-1]
            ax_map.set_ylim([2, y])
            prof = dPow_list[:y].sum(axis=0)
            plot.set_ydata(prof)
            ax_prof.set_ylim([prof.min(), prof.max()])    
        fig.canvas.draw()
    _ = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
    return lim_list[-1]

def _get_TP(series):
    """
    Get the number of local max and min in a curve.
    """
  
    Ntp = np.sum(np.absolute(np.diff(np.sign(np.diff(series)))) == 2)
    N = series.size
    sig = np.sqrt((16 * N - 29) / 90.)
    E = 2 * (N - 2) / 3.
    z = (Ntp - E) / sig
    return np.abs(z)
    
def _Poly_Max(x, y, Err):
    """
    Polynomial fit
    """
    
    Pows= np.arange(2, y.size - 1, dtype=float)
    Met = np.zeros_like(Pows)
    SN  = np.zeros_like(Pows)
    for i, deg in enumerate(Pows):
        fit = np.polyfit(x, y, deg)
        Res = y - np.polyval(fit, x)
        SN[i] = _get_TP(Res)    
        Met[i] = np.std(Res)
    n   = np.where(Met == Met[SN == SN.min()].min())[0][0] + 2
    p = np.polyfit(x, y, n)
    Fac = np.std(y) / Err
    
    dp      = np.polyder(p)
    ddp     = np.polyder(dp)
    cands   = np.roots(dp)
    r_cands = np.polyval(ddp, cands)
    first_cut = cands[(cands.imag==0) & (cands.real>=min(x)) & (cands.real<=max(x)) & (r_cands<0)]
    if first_cut.size > 0:
        Value     = np.polyval(p, first_cut)
        Best      = first_cut[Value.argmax()]
        delta_x   = np.sqrt(np.abs(2 * Err / np.polyval(ddp, Best)))
    else:
        Best    = 0
        delta_x = 0
    
    return np.real(Best), delta_x, p , Fac
  
def _plot_Power(DM_Map, X, Y, Range, Returns_Poly, x, y, SN, filename=""):
    """
    Diagnostic plot of Coherent Power vs Dispersion Measure
    """
    
    fig = plt.figure(figsize=(6, 8.5), facecolor='k')
    fig.subplots_adjust(left=0.1, bottom=0.05, right=0.99, top=0.88)
    gs = gridspec.GridSpec(3, 1, hspace=0, height_ratios=[3, 1, 9])
    ax_prof = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_prof)
    ax_map = fig.add_subplot(gs[2], sharex=ax_prof)
    
    Title = '{0:}\n\
        Best DM = {1:.3f} $\pm$ {2:.3f}\n\
        S/N = {3:.1f}'.format(filename, Returns_Poly[0], Returns_Poly[1], SN)
    fig.suptitle(Title, color='w', linespacing=1.5) 
    
    # Profile
    ax_prof.plot(X, Y, 'w-', linewidth=3, clip_on=False)
    ax_prof.plot(X[Range], np.polyval(Returns_Poly[2], X[Range]), color='orange', linewidth=3, zorder=2, clip_on=False) 
    ax_prof.set_xlim([X.min(), X.max()])
    ax_prof.set_ylim([Y.min(), Y.max()])
    ax_prof.axis('off')
        
    # Residuals
    Res = y - np.polyval(Returns_Poly[2], x)
    Res -= Res.min()
    Res /= Res.max()
    ax_res.plot(x, Res, 'xw', linewidth=2, clip_on=False)
    ax_res.set_ylim([np.min(Res) - np.std(Res) / 2, np.max(Res) + np.std(Res) / 2])
    ax_res.set_ylabel('$\Delta$')
    ax_res.tick_params(axis='both', colors='w', labelbottom='off', labelleft='off', direction='in', left='off', top='on')
    ax_res.yaxis.label.set_color('w')
    ax_res.set_facecolor('k')

    # Power vs DM map      
    FT_len = DM_Map.shape[0]
    ar = psrchive.Archive_load(filename)
    dt = ar.get_first_Integration().get_duration()
    Bin_len = ar.get_nbin()
    extent = [np.min(X), np.max(X), 0, 2 * np.pi * FT_len / Bin_len / dt]
    ax_map.imshow(DM_Map, origin='lower', aspect='auto', cmap='YlOrBr_r', extent=extent, interpolation='spline16')
    ax_map.tick_params(axis='both', colors='w', direction='in', right='on', top='on')
    ax_map.xaxis.label.set_color('w')
    ax_map.yaxis.label.set_color('w')
    ax_map.set_xlabel('DM (pc / cc)')
    ax_map.set_ylabel('w (rad / ms)')
    try: fig.align_ylabels([ax_map, ax_res])  #Recently added feature
    except AttributeError:
        ax_map.yaxis.set_label_coords(-0.07, 0.5)
        ax_res.yaxis.set_label_coords(-0.07, 0.5)

    fig.savefig(filename, facecolor='k', edgecolor='k')
    return  

def _get_Window(Pro):
    """
    ACF Windowing
    """
    
    arr = scipy.signal.detrend(Pro)
    X = np.correlate(arr, arr, "same")
    n = X.argmax()
    W = np.max(np.diff(np.where(X < 0)))
    return W

def _check_W(Pro, W):
    """
    Check whether the veiwing window will be in the index range.
    """

    SM = np.convolve(Pro, np.ones(W), 'same')
    Peak = np.mean(np.where(SM == max(SM)))
    Max = np.where(Pro == np.max(Pro))
    if (Peak - Max)**2 > W**2:
        W += np.abs(Peak - Max) / 2
        Peak = (Peak + Max) / 2
    Start = np.int(Peak - np.round(1.25 * W))
    End = np.int(Peak + np.round(1.25 * W))
    if Start < 0: Start=0
    if End > Pro.size - 1: End = Pro.size - 1
    return Start,End

def _plot_waterfall(Returns_Poly, waterfall, dt, f, Cut_off, fname=""):
    """
    Plot the waterfall at the best Dispersion Measure and at close values for comparison. 
    """
    
    fig = plt.figure(figsize=(8.5, 6), facecolor='k')
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.99, top=0.8)
    grid = gridspec.GridSpec(1, 3, wspace=0.1)
    
    Title='{0:}\n\
        Best DM = {1:.3f} $\pm$ {2:.3f}'.format(fname, Returns_Poly[0], Returns_Poly[1])
    plt.suptitle(Title, color='w', linespacing=1.5)
    
    DMs = Returns_Poly[0] + 5 * Returns_Poly[1] * np.array([-1, 0, 1])  # DMs +- 5 sigmas away
    for j, dm in enumerate(DMs):
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[j], height_ratios=[1, 4], hspace=0)
        ax_prof = fig.add_subplot(gs[0])
        ax_wfall = fig.add_subplot(gs[1], sharex=ax_prof)
        
        wfall = _dedisperse_waterfall(waterfall, dm)
        prof = wfall.sum(axis=0)
        
        # Find the time range around the pulse
        if j==0: 
            W = _get_Window(prof)
            Spect = _get_Spect(wfall)
            Filter = np.ones_like(Spect)
            Filter[Cut_off : -Cut_off] = 0
            Spike = np.real(ifft(Spect * Filter))
            Spike[0] = 0
            Win = _check_W(Spike, W)

        # Profile
        T = dt * (Win[1] - Win[0])
        x = np.linspace(0, T, Win[1] - Win[0])
        y = prof[Win[0] : Win[1]]
        ax_prof.plot(x, y, 'w', linewidth=0.5, clip_on=False)
        ax_prof.axis('off')
        ax_prof.set_title('{0:.3f}'.format(dm), color='w')

        # Waterfall
        bw = f[-1] - f[0]
        im = wfall[:, Win[0] : Win[1]]
        extent = [0, T, f[0], f[-1]]
        MAX_DS = wfall.max()
        MIN_DS = wfall.mean() - wfall.std()
        ax_wfall.imshow(im, origin='lower', aspect='auto', cmap='YlOrBr_r', extent=extent, interpolation='nearest', vmin=MIN_DS, vmax=MAX_DS)

        ax_wfall.tick_params(axis='both', colors='w', direction='in', right='on', top='on')
        if j == 0: ax_wfall.set_ylabel('Frequency (MHz)')
        if j == 1: ax_wfall.set_xlabel('Time (ms)')
        if j > 0: ax_wfall.tick_params(axis='both', labelleft='off')
        ax_wfall.yaxis.label.set_color('w')
        ax_wfall.xaxis.label.set_color('w')

    fig.savefig(fname, facecolor='k', edgecolor='k')
    return

def _dedisperse_waterfall(wfall, DM, freq):
    """
    Dedisperse a wfall matrix to DM.
    """
    
    k_DM = 
    dedisp = np.zeros_like(wfall)
    for i,ts in wfall:
        n = k_DM * DM * (freq[-1]**-2 - freq[i]**-2)
        dedisp[i] = np.roll(ts, n)
    return dedisp
    
def _init_DM(archive, DM_s, DM_e):
    """
    Initialize DM limits of the search if not specified.
    """
    
    DM = archive.get_dm()
    if DM_s is None: DM_s = DM - 10
    if DM_e is None: DM_e = DM + 10
    return DM_s, DM_e

def from_PSRCHIVE(fname, DM_s, DM_e, DM_step, manual_cutoff=False):
    """
    Brute-force search of the Dispersion Measure of a single pulse stored into a PSRCHIVE file.
    The algorithm uses phase information and is robust to interference and unusual burst shapes.

    Parameters
    ----------
    fname : str
        Name of a PSRCHIVE file.
    DM_s : float
        Starting value of Dispersion Measure to search (pc/cc).
    DM_e : float
        Ending value of Dispersion Measure to search (pc/cc).
    DM_step : float
        Step of the search (pc/cc).
        
    Returns
    -------
    DM : float
        Best value of Dispersion Measure (pc/cc).
    DM_std :
        Standard deviation of the Dispersion Measure (pc/cc)
        
    Stores
    ------
    basename(fname) + "_Waterfall_5sig.pdf" : plot
        Pulse waterfall at the best Dispersion Measure and 5 sigmas away
    basename(filename) + "_DM_Search.pdf": plot
        Map of the coherent power as a function of the search Dispersion Measure.
    """
    
    waterfall, f_channels, t_res = _load_psrchive(fname)
    DM_s, DM_e = _init_DM(archive, DM_s, DM_e)
    DM_list = np.arange(np.float(DM_s), np.float(DM_e), np.float(DM_step))
    DM, DM_std = get_DM(waterfall, DM_list, t_res, f_channels, manual_cutoff=manual_cutoff, fname=os.path.basename(fname))
    return DM, DM_std

def get_DM(waterfall, DM_list, t_res, f_channels, manual_cutoff=False, diagnostic_plots=True, fname=""):
    """
    Brute-force search of the Dispersion Measure of a waterfall numpy matrix.
    The algorithm uses phase information and is robust to interference and unusual burst shapes.

    Parameters
    ----------
    waterfall : ndarray
        2D array with shape (frequency channels, phase bins)
    DM_list : list
        List of Dispersion Measure values to search (pc/cc).
    t_res : float
        Time resolution of each phase bin (s).
    f_channels : list
        Central frequency of each channel, from low to high (MHz).
    manual_cutoff : bool, optional. Default = False
        If False, the power spectrum cutoff is automatically selected.
    diagnostic_plots : bool, optional. Default = True
        Stores the diagnostic plots "Waterfall_5sig.pdf" and "DM_Search.pdf"
    fname : str, optional. Default = ""
        Filename used as a prefix for the diagnostic plots.
    
    Returns
    -------
    DM : float
        Best value of Dispersion Measure (pc/cc).
    DM_std :
        Standard deviation of the Dispersion Measure (pc/cc)
    """

    nchan = waterfall.shape[0]
    nbin = waterfall.shape[1] / 2
    Pow_list = np.zeros([nbin, DM_list.size])
    for i, DM in enumerate(DM_list):
        waterfall_dedisp = _dedisperse_waterfall(matrix, DM, f_channels)
        Pow = _get_Pow(waterfall_dedisp)
        Pow_list[:, i] = Pow[: nbin]
    
    v = np.arange(0, nbin)
    dPow_list = Pow_list * v[:, np.newaxis]**2
    
    Mean     = nchan  # Base on Gamma(2,)
    STD      = Mean / np.sqrt(2)  # Base on Gamma(2,)
    if manual_cutoff: fact_idx = _get_f_threshold_manual(Pow_list, dPow_list)
    else: fact_idx = _get_f_threshold(Pow_list, Mean, STD)
    DM_curve = dPow_list[:fact_idx].sum(axis=0)

    Max   = DM_curve.max()
    dMean = fact_idx * (fact_idx + 1) * (2 * fact_idx + 1) / 6. * Mean
    dSTD  = STD * np.sqrt(fact_idx * (fact_idx + 1) * (2 * fact_idx + 1) * (3 * fact_idx**2 + 3 * fact_idx - 1) / 30.)
    SN    = (Max - dMean) / dSTD
    
    Peak  = DM_curve.argmax()
    Range = np.arange(Peak - 2, Peak + 2)
    y = DM_curve[Range]
    x = DM_list[Range]
    Returns_Poly = _Poly_Max(x, y, dSTD)
    
    if fname != "": fname += "_"
    _plot_Power(Pow_list[:fact_idx], DM_list, DM_curve, Range, Returns_Poly, x, y, SN, fname=fname+"DM_Search.pdf")
    _plot_waterfall(Returns_Poly, waterfall, t_res, f_channels, fact_idx, fname=fname+"Waterfall_5sig.pdf")
    
    DM = Returns_Poly[0]
    DM_std = Returns_Poly[1]
    return DM, DM_std

def _get_parser():
    """
    Argument parser.
    """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Search for best DM based on FFT phase angles.")
    parser.add_argument('fname', help="Filename of the PSRCHIVE file.")
    parser.add_argument('-DM_s', help="Start DM. If None, it will select the DM from the PSRCHIVE file.", default=None, type=float)
    parser.add_argument('-DM_e', help="End DM. If None, it will select the DM from the PSRCHIVE file.", default=None, type=float)
    parser.add_argument('-DM_step', help="Step DM.", default=0.1, type=float)
    parser.add_argument('-manual_cutoff', help="Manually set the FFT frequency cutoff.", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_parser()
    DM, DM_std = from_PSRCHIVE(args.fname, args.DM_s, args.DM_e, args.DM_step, manual_cutoff=args.manual_cutoff)


