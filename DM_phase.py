"""
Incoherent search for best Dispersion Measure from a PSRCHIVE file.
The search uses phase information and thus it is not sensitive to Radio Frequency Interference or complex spectro-temporal pulse shape.
"""

import os
import argparse
import sys
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Cursor, SpanSelector, Button
import scipy.signal
from scipy.fftpack import fft, ifft


plt.rcParams['toolbar'] = 'None'
plt.rcParams['keymap.yscale'] = 'Y'
colormap_list = cycle(['YlOrBr_r', 'viridis', 'Greys'])
colormap = next(colormap_list)

def _load_psrchive(fname):
    """
    Load data from a PSRCHIVE file.
    """
    
    archive = psrchive.Archive_load(fname)
    archive.pscrunch()
    archive.set_dispersion_measure(0.)  # Un-dedisperse
    archive.dedisperse()
    archive.set_dedispersed(False)
    archive.tscrunch()
    archive.centre()
    w = archive.get_weights().squeeze()
    waterfall = np.ma.masked_array(archive.get_data().squeeze())
    #waterfall *= w[:, np.newaxis]
    waterfall[w == 0] = np.ma.masked
    f_ch = np.array([archive.get_first_Integration().get_centre_frequency(i) for i in range(archive.get_nchan())])
    dt = archive.get_first_Integration().get_duration() / archive.get_nbin()
    
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
    Kern = np.round( _get_Window(SN)/2 ).astype(int)
    if Kern < 5: Kern = 5
    return 0, Kern

def _get_f_threshold_manual(Pow_list, dPow_list, waterfall, DM_list, f_channels, t_res):
    """
    Select power limits with interactive GUI.
    """
    
    # Define axes
    fig = plt.figure(figsize=(12., 8.5), facecolor='k')
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.95, top=0.94, hspace=0)
    gs = gridspec.GridSpec(2, 3, hspace=0, wspace=0.02, height_ratios=[1, 4], width_ratios=[2, 3, 2])
    ax_text     = fig.add_subplot(gs[1, 0])
    ax_pow_prof = fig.add_subplot(gs[0, 1])
    ax_pow_map  = fig.add_subplot(gs[1, 1], sharex=ax_pow_prof)
    ax_wat_prof = fig.add_subplot(gs[0, 2])
    ax_wat_map  = fig.add_subplot(gs[1, 2], sharex=ax_wat_prof)
    
    for ax in fig.axes:
        ax.axis('off')
    
    # Plot power
    plot_pow_map = ax_pow_map.imshow(Pow_list, origin='lower', aspect='auto', cmap=colormap, interpolation='nearest')
    ax_pow_map.set_ylim([0, Pow_list.shape[0]])
    pow_prof = dPow_list.sum(axis=0)
    plot_pow_prof, = ax_pow_prof.plot(pow_prof, 'w-', linewidth=2, clip_on=False)
    ax_pow_prof.set_ylim([pow_prof.min(), pow_prof.max()])
    ax_pow_prof.set_title('Coherent power', fontsize=16, color='w', y=1.08)
    
    # Plot waterfall
    top_lim = [Pow_list.shape[0],]
    bottom_lim = [0,]
    DM, _ = _DM_calculation(waterfall, Pow_list, dPow_list, bottom_lim[-1], top_lim[-1], f_channels, t_res, DM_list, no_plots=True)
    waterfall_dedisp = _dedisperse_waterfall(waterfall, DM, f_channels, t_res)
    plot_wat_map = ax_wat_map.imshow(waterfall_dedisp, origin='lower', aspect='auto', cmap=colormap, interpolation='nearest')
    wat_prof = waterfall_dedisp.sum(axis=0)
    plot_wat_prof, = ax_wat_prof.plot(wat_prof, 'w-', linewidth=2)
    ax_wat_prof.set_ylim([wat_prof.min(), wat_prof.max()])
    ax_wat_prof.set_xlim([0, wat_prof.size])
    ax_wat_prof.set_title("Waterfall", fontsize=16, color='w', y=1.08)
    
    # Plot instructions
    text = """
    Manual selection of
      power limits.
      
    Current best DM = {:0.2f}
    
    On the left plot, press
      "t" to select top limit.
      "b" to select bottom limit.
      "T" to undo upper limit.
      "B" to undo lower limit.
      "l" for logarithmic scale.
      "q" to save and exit.
      
    On the right plot, 
      drag mouse to zoom in.
      space bar to reset zoom.
      
    """
    instructions = ax_text.annotate(text.format(DM), (0, 1), color='w', fontsize=14, horizontalalignment='left', verticalalignment='top', linespacing=1.5)

    # GUI
    def update_lim(is_log):
        if is_log: pow_map = np.log10(Pow_list[bottom_lim[-1] : top_lim[-1]])
        else: pow_map = Pow_list[bottom_lim[-1] : top_lim[-1]]
        plot_pow_map.set_clim(vmin=pow_map.min(), vmax=pow_map.max())
        ax_pow_map.set_ylim([bottom_lim[-1], top_lim[-1]])
        pow_prof = dPow_list[bottom_lim[-1] : top_lim[-1]].sum(axis=0)
        plot_pow_prof.set_ydata(pow_prof)
        ax_pow_prof.set_ylim([pow_prof.min(), pow_prof.max()])
        DM, _ = _DM_calculation(waterfall, Pow_list, dPow_list, bottom_lim[-1], top_lim[-1], f_channels, t_res, DM_list, no_plots=True)
        waterfall_dedisp = _dedisperse_waterfall(waterfall, DM, f_channels, t_res)
        plot_wat_map.set_data(waterfall_dedisp)
        wat_prof = waterfall_dedisp.sum(axis=0)
        plot_wat_prof.set_ydata(wat_prof)
        ax_wat_prof.set_ylim([wat_prof.min(), wat_prof.max()])
        instructions.set_text(text.format(DM))
        return 
    
    is_log = [False]
    def press(event):
        sys.stdout.flush()
        if event.key == "t":
            y = int(round(event.ydata))
            top_lim.append(y)
            update_lim(is_log[0])
        if event.key == "b":
            y = int(round(event.ydata))
            bottom_lim.append(y)
            update_lim(is_log[0])
        elif event.key == "T":
            if len(top_lim) > 1: del top_lim[-1]
            update_lim(is_log[0])
        elif event.key == "B":
            if len(bottom_lim) > 1: del bottom_lim[-1]
            update_lim(is_log[0])
        elif event.key == "l":
            if is_log[0]:
                plot_pow_map.set_data(Pow_list)
                plot_pow_map.set_clim(vmin=Pow_list[bottom_lim[-1] : top_lim[-1]].min(), vmax=Pow_list[bottom_lim[-1] : top_lim[-1]].max()) 
                is_log[0] = False
            else:
                Pow_list_log = np.log10(Pow_list)
                plot_pow_map.set_data(Pow_list_log)
                plot_pow_map.set_clim(vmin=Pow_list_log[bottom_lim[-1] : top_lim[-1]].min(), vmax=Pow_list_log[bottom_lim[-1] : top_lim[-1]].max()) 
                is_log[0] = True
        elif event.key == " ": 
            ax_wat_prof.set_xlim([0, wat_prof.size])
            xlim[0] = 0
            xlim[1] = wat_prof.size
        fig.canvas.draw()
        return
    
    xlim = [0, wat_prof.size]
    def onselect_prof(xmin, xmax):
        ax_wat_prof.set_xlim(xmin, xmax)
        xlim[0] = int(xmin)
        xlim[1] = int(xmax)
        fig.canvas.draw()
        return
        
    def onselect_map(xmin, xmax):
        ax_wat_prof.set_xlim(xmin, xmax)
        xlim[0] = int(xmin)
        xlim[1] = int(xmax)
        fig.canvas.draw()
        return

    def new_cmap(event):
        colormap = next(colormap_list)
        plot_wat_map.set_cmap(colormap)
        plot_pow_map.set_cmap(colormap)
        fig.canvas.draw()
        return

    ax_but = plt.axes([0.01, 0.94, 0.12, 0.05])
    but = Button(ax_but, 'Change colormap', color='0.8', hovercolor='0.2')
    but.on_clicked(new_cmap)
    span_prof = SpanSelector(ax_wat_prof, onselect_prof, 'horizontal', rectprops=dict(alpha=0.5, facecolor='g'))
    span_map = SpanSelector(ax_wat_map, onselect_map, 'horizontal', rectprops=dict(alpha=0.5, facecolor='g'))
    try: cursor = Cursor(ax_pow_map, color='g', linewidth=2, vertOn=False)
    except AttributeError: pass
    key = fig.canvas.mpl_connect('key_press_event', press)

    plt.show()
    return bottom_lim[-1], top_lim[-1], xlim

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
    n = np.linalg.matrix_rank(np.vander(y))
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
        Best    = 0.
        delta_x = 0.
    
    return float(np.real(Best)), delta_x, p , Fac
  
def _plot_Power(DM_Map, low_idx, up_idx, X, Y, Range, Returns_Poly, x, y, SN, t_res, fname=""):
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
        S/N = {3:.1f}'.format(fname, Returns_Poly[0], Returns_Poly[1], SN)
    fig.suptitle(Title, color='w', linespacing=1.5) 
    
    # Profile
    ax_prof.plot(X, Y, 'w-', linewidth=3, clip_on=False)
    ax_prof.plot(X[Range], np.polyval(Returns_Poly[2], X[Range]), color='orange', linewidth=3, zorder=2, clip_on=False) 
    ax_prof.set_xlim([X.min(), X.max()])
    ax_prof.set_ylim([Y.min(), Y.max()])
    ax_prof.axis('off')
    ax_prof.ticklabel_format(useOffset=False)
        
    # Residuals
    Res = y - np.polyval(Returns_Poly[2], x)
    Res -= Res.min()
    Res /= Res.max()
    ax_res.plot(x, Res, 'xw', linewidth=2, clip_on=False)
    ax_res.set_ylim([np.min(Res) - np.std(Res) / 2, np.max(Res) + np.std(Res) / 2])
    ax_res.set_ylabel('$\Delta$')
    ax_res.tick_params(axis='both', colors='w', labelbottom='off', labelleft='off', direction='in', left='off', top='on')
    ax_res.yaxis.label.set_color('w')
    try: ax_res.set_facecolor('k')
    except AttributeError: ax_res.set_axis_bgcolor('k')
    ax_res.ticklabel_format(useOffset=False)
    
    # Power vs DM map      
    FT_len = DM_Map.shape[0]
    indx2Ang = 1. / (2 * FT_len * t_res * 1000)
    extent = [np.min(X), np.max(X), low_idx * indx2Ang, up_idx * indx2Ang]
    ax_map.imshow(DM_Map[low_idx : up_idx], origin='lower', aspect='auto', cmap=colormap, extent=extent, interpolation='nearest')
    ax_map.tick_params(axis='both', colors='w', direction='in', right='on', top='on')
    ax_map.xaxis.label.set_color('w')
    ax_map.yaxis.label.set_color('w')
    ax_map.set_xlabel('DM (pc cm$^{-3}$)')
    ax_map.set_ylabel('Fluctuation Frequency (ms$^{-1}$)')  #From p142 in handbook, also see Camilo et al. (1996)
    ax_map.ticklabel_format(useOffset=False)
    try: fig.align_ylabels([ax_map, ax_res])  #Recently added feature
    except AttributeError:
        ax_map.yaxis.set_label_coords(-0.07, 0.5)
        ax_res.yaxis.set_label_coords(-0.07, 0.5)
        
    if fname != "": fname += "_"
    fig.savefig(fname + "DM_Search.pdf", facecolor='k', edgecolor='k')
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

def _plot_waterfall(Returns_Poly, waterfall, dt, f, Cut_off, fname="", Win=None):
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
        try: ax_wfall.set_facecolor('k')
        except AttributeError: ax_wfall.set_axis_bgcolor('k')
        
        wfall = _dedisperse_waterfall(waterfall, dm, f, dt)
        prof = wfall.sum(axis=0)
        
        # Find the time range around the pulse
        if (j == 0) and (Win is None):
            W = _get_Window(prof)
            Spect = _get_Spect(wfall)
            Filter = np.ones_like(Spect)
            Filter[Cut_off : -Cut_off] = 0
            Spike = np.real(ifft(Spect * Filter))
            Spike[0] = 0
            Win = _check_W(Spike, W)

        # Profile
        T = dt * (Win[1] - Win[0]) * 1000
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
        ax_wfall.imshow(im, origin='lower', aspect='auto', cmap=colormap, extent=extent, interpolation='nearest', vmin=MIN_DS, vmax=MAX_DS)

        ax_wfall.tick_params(axis='both', colors='w', direction='in', right='on', top='on')
        if j == 0: ax_wfall.set_ylabel('Frequency (MHz)')
        if j == 1: ax_wfall.set_xlabel('Time (ms)')
        if j > 0: ax_wfall.tick_params(axis='both', labelleft='off')
        ax_wfall.yaxis.label.set_color('w')
        ax_wfall.xaxis.label.set_color('w')

    if fname != "": fname += "_"
    fig.savefig(fname + "Waterfall_5sig.pdf", facecolor='k', edgecolor='k')
    return

def _dedisperse_waterfall(wfall, DM, freq, dt):
    """
    Dedisperse a wfall matrix to DM.
    """
    
    k_DM = 1. / 2.41e-4
    dedisp = np.zeros_like(wfall)
    shift = (k_DM * DM * (freq[-1]**-2 - freq**-2) / dt).round().astype(int)
    for i,ts in enumerate(wfall):
        dedisp[i] = np.roll(ts, shift[i])
    return dedisp
    
def _init_DM(fname, DM_s, DM_e):
    """
    Initialize DM limits of the search if not specified.
    """
    
    archive = psrchive.Archive_load(fname)
    DM = archive.get_dispersion_measure()
    if DM_s is None: DM_s = DM - 10
    if DM_e is None: DM_e = DM + 10
    return DM_s, DM_e

def from_PSRCHIVE(fname, DM_s, DM_e, DM_step, manual_cutoff=False, no_plots=False):
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
    basename(fname) + "_DM_Search.pdf": plot
        Map of the coherent power as a function of the search Dispersion Measure.
    """
    
    waterfall, f_channels, t_res = _load_psrchive(fname)
    DM_s, DM_e = _init_DM(fname, DM_s, DM_e)
    DM_list = np.arange(np.float(DM_s), np.float(DM_e), np.float(DM_step))
    DM, DM_std = get_DM(waterfall, DM_list, t_res, f_channels, manual_cutoff=manual_cutoff, fname=os.path.basename(fname), no_plots=no_plots)
    return DM, DM_std

def get_DM(waterfall, DM_list, t_res, f_channels, manual_cutoff=False, diagnostic_plots=True, fname="", no_plots=False):
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
        waterfall_dedisp = _dedisperse_waterfall(waterfall, DM, f_channels, t_res)
        Pow = _get_Pow(waterfall_dedisp)
        Pow_list[:, i] = Pow[: nbin]

    v = np.arange(0, nbin)
    dPow_list = Pow_list * v[:, np.newaxis]**2
    
    Mean     = nchan               # Base on Gamma(2,)
    STD      = nchan / np.sqrt(2)  # Base on Gamma(2,)
    if manual_cutoff: low_idx, up_idx, phase_lim = _get_f_threshold_manual(Pow_list, dPow_list, waterfall, DM_list, f_channels, t_res)
    else: 
        low_idx, up_idx = _get_f_threshold(Pow_list, Mean, STD)
        phase_lim = None
    
    DM, DM_std = _DM_calculation(waterfall, Pow_list, dPow_list, low_idx, up_idx, f_channels, t_res, DM_list, no_plots=no_plots, fname=fname, phase_lim=phase_lim)
    return DM, DM_std
    
def _DM_calculation(waterfall, Pow_list, dPow_list, low_idx, up_idx, f_channels, t_res, DM_list, no_plots=False, fname="", phase_lim=None):
    """
    Calculate the best DM value.
    """

    DM_curve = dPow_list[low_idx : up_idx].sum(axis=0)

    fact_idx = up_idx - low_idx
    Max   = DM_curve.max()
    nchan = len(f_channels)
    Mean  = nchan              # Base on Gamma(2,)
    STD   = Mean / np.sqrt(2)  # Base on Gamma(2,)
    m_fact = np.sum(np.arange(low_idx, up_idx)**2)
    s_fact = np.sum(np.arange(low_idx, up_idx)**4)**0.5
    dMean = Mean * m_fact
    dSTD  = STD  * s_fact 
    SN    = (Max - dMean) / dSTD
    
    Peak  = DM_curve.argmax()
    Range = np.arange(Peak - 5, Peak + 5)
    y = DM_curve[Range]
    x = DM_list[Range]
    Returns_Poly = _Poly_Max(x, y, dSTD)
    
    if not no_plots:
        _plot_Power(Pow_list, low_idx, up_idx, DM_list, DM_curve, Range, Returns_Poly, x, y, SN, t_res, fname=fname)
        _plot_waterfall(Returns_Poly, waterfall, t_res, f_channels, fact_idx, fname=fname, Win=phase_lim)
    
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
    parser.add_argument('-manual', help="Manually set the FFT frequency cutoff.", action='store_true')
    parser.add_argument('-no_plots', help="Do not produce diagnostic plots.", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_parser()
    import psrchive
    DM, DM_std = from_PSRCHIVE(args.fname, args.DM_s, args.DM_e, args.DM_step, manual_cutoff=args.manual, no_plots=args.no_plots)
