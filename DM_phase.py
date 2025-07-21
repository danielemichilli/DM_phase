"""Incoherent search for best dispersion measure from a PSRCHIVE file.

The search uses phase information and thus it is not sensitive to Radio
Frequency Interference or complex spectro-temporal pulse structure.

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
from scipy.stats import norm


plt.rcParams['toolbar'] = 'None'
plt.rcParams['keymap.yscale'] = 'Y'

COLORMAP_LIST = cycle(['YlOrBr_r', 'viridis', 'Greys'])
COLORMAP = next(COLORMAP_LIST)


def _load_psrchive(fname):
    """Load data from a PSRCHIVE file.

    Parameters
    ----------
    fname : str
        Archive (.ar) to load.

    Returns
    -------
    waterfall : array_like
        Burst dynamic spectrum.
    f_channels : array_like
        Center frequencies, in MHz.
    t_res : float
        Sampling time, in s.

    """
    archive = psrchive.Archive_load(fname)
    archive.pscrunch()
    # un-dedisperse
    archive.set_dispersion_measure(0.)
    archive.dedisperse()
    archive.set_dedispersed(False)
    archive.tscrunch()
    archive.centre()
    weights = archive.get_weights().squeeze()
    waterfall = np.ma.masked_array(archive.get_data().squeeze())
    waterfall[weights == 0] = np.ma.masked
    f_channels = np.array([
        archive.get_first_Integration().get_centre_frequency(i) \
        for i in range(archive.get_nchan())])
    t_res = archive.get_first_Integration().get_duration() \
        / archive.get_nbin()

    if archive.get_bandwidth() < 0:
        waterfall = np.flipud(waterfall)
        f_channels = f_channels[::-1]

    return waterfall, f_channels, t_res


def _get_coherence_spectrum(waterfall):
    """Get the 'coherence' spectrum of a waterfall, by taking a
    one-dimensional Fourier transform of the intensity data along the
    frequency channels, dividing by the amplitude (thus keeping only the
    phase information) and summing over the emission bandwidth.

    Parameters
    ----------
    waterfall : array_like
        Burst dynamic spectrum.

    Returns
    -------
    coherence_spectrum : array_like
        'Coherence' spectrum.

    """
    fourier_transform = fft(waterfall, axis=-1)
    amp = np.abs(fourier_transform)
    amp[amp == 0] = 1
    coherence_spectrum = np.sum(fourier_transform / amp, axis=0)

    return coherence_spectrum


def _get_coherent_power_spectrum(waterfall):
    """Get the 'coherence' power spectrum of a waterfall.

    Parameters
    ----------
    waterfall : array_like
        Burst dynamic spectrum.

    Returns
    -------
    power_spectrum : array_like
        'Coherence' power spectrum.

    """
    coherence_spectrum = _get_coherence_spectrum(waterfall)
    power_spectrum = np.abs(coherence_spectrum) ** 2

    return power_spectrum


def _get_f_threshold(power_spectra, mean, std):
    """Get the Fourier (fluctuation) frequency cutoff.

    Parameters
    ----------
    power_spectra : array_like
        'Coherence' power spectra for a range of trial DMs.
    mean : int
        Expectation value of the power distribution; should be the
        number of frequency channels.
    std : float
        Standard deviation of the power distribution; should be the
        square-root of the number of channels divided by 2.

    Returns
    -------
    tuple of int
        Minimum and maximum fluctuation frequency index to consider.

    """
    peak_power = np.max(power_spectra, axis=1)
    snr = (peak_power - mean) / std
    kern = np.round(_get_window(snr) / 2).astype(int)
    # always use at least 5
    if kern < 5:
        kern = 5

    return 0, kern


def _get_dm_curve(power_spectra, dpower_spectra,nchan):
    """Get the integrated fluctuation frequency for each spectrum.
    The cutoff for each spectrum is calculated by minimizing the variance of the noise."""

    n = power_spectra.shape[0]
    m = power_spectra.shape[1]
    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    num_el = (n - Y).astype(float)
    S = np.divide(
        np.sum(power_spectra, axis=0).T - np.cumsum(power_spectra, axis=0),
        num_el
    )
    S2 = np.divide(
        np.sum(power_spectra**2, axis=0).T - np.cumsum(power_spectra**2, axis=0),
        num_el
    )
    var = np.divide( (S2 - S**2), num_el)
    var_sm = scipy.signal.convolve2d(var, np.ones([9, 3]) /27 , mode='same', boundary='wrap')
    idx_f = np.argmin(var_sm[:-10, :], axis=0)
    idx_c = np.convolve(idx_f, np.ones(3) / 3., mode='same').astype(int)
    idx_c[idx_c==0] = 1
    idx_c = np.ones(np.shape(idx_c))*(idx_c)
    I = np.ones([n, 1]) * idx_c
    I2_sum = np.multiply(np.multiply(idx_c,idx_c+1),2*idx_c+1)/6
    I4_sum = np.multiply(np.multiply(np.multiply(idx_c,idx_c+1),2*idx_c+1),3*idx_c+3*idx_c-1)/30

    Lo = np.multiply(Y <= I, dpower_spectra)
    Lo1= np.multiply(Y <= I, np.multiply(power_spectra,dpower_spectra))
    AV_N_pow = 2.0*nchan*np.ones( np.shape(idx_c) )
    dm_curve = Lo.sum(axis=0)
    dn_term  = Lo1.sum(axis=0)
    Noise_curve = np.multiply( AV_N_pow, I2_sum )
    #Dem = (Noise_curve/n/2 )
    #Dem = ( nchan * idx_c * n  / np.pi**2)
    #Dem = nchan/3/n *idx_c**3
    Var_dp  = ( 2.0* nchan**2 *I4_sum + 1.0*dn_term )
    dm_c_err = Var_dp**0.5
    Dem  = ( 2.0* nchan**2 * ( I4_sum + 2.0 * I2_sum) )**0.5
    SN =  np.divide( ( dm_curve - 1.0*Noise_curve ), Dem )
    #SN_Err = ( 1 + 2*nchan**2 * (1.0*I4_sum + 2.0*I2_sum) / Dem**2)**0.5
    SN_Err = (np.divide(Var_dp,Dem**2) + 1 + np.multiply( np.divide(SN**2,idx_c), (1 + 8*nchan**2 / Dem**2 ) ) )**0.5
    #SN_Err = np.divide(SN_Err,SN)
    SN[np.isnan(SN)]=0.

    return dm_curve, dm_c_err, SN


def _get_frequency_range_manual(waterfall, f_channels):
    """Select frequency range to use with GUI.

    Parameters
    ----------
    waterfall : array_like
        Burst dynamic spectrum.
    f_channels : array_like
        Center frequencies, in MHz.

    Returns
    -------
    tuple of int
        Minimum and maximum emission frequency index to consider.

    """
    bg_color = "k"
    fg_color = "w"

    fig = plt.figure(figsize=(8., 8.5), facecolor=bg_color)
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.95, top=0.94, hspace=0)
    gs = gridspec.GridSpec(2, 2, hspace=0, height_ratios=[1, 4],
                           width_ratios=[2, 2])
    ax_text = fig.add_subplot(gs[1, 0])
    ax_wat_prof = fig.add_subplot(gs[0, 1])
    ax_wat_map = fig.add_subplot(gs[1, 1], sharex=ax_wat_prof)

    for ax in fig.axes:
        ax.axis('off')

    ax_wat_map.axis('on')
    ax_wat_map.spines['left'].set_color(fg_color)
    ax_wat_map.tick_params(axis='y', colors=fg_color)
    ax_wat_map.yaxis.label.set_color(fg_color)

    # plot waterfall
    top_lim = [waterfall.shape[0],]
    bottom_lim = [0,]
    sub_factor = [1,]
    left = 0
    right = waterfall.shape[1]

    plot_wat_map = ax_wat_map.imshow(waterfall, origin='lower', aspect='auto',
                                     cmap=COLORMAP, interpolation='none',
                                     extent=(left, right, bottom_lim[-1] - 0.5,
                                             top_lim[-1] + 0.5))

    ax_wat_map.set_ylabel("Observing frequency (MHz)", fontsize=14)
    df = np.median(np.diff(f_channels))

    # set frequencies as label instead of channel numbers
    yticks = np.linspace(bottom_lim[-1] - 0.5, top_lim[-1] + 0.5, 9)
    yticklabels = np.round(np.linspace(f_channels[bottom_lim[-1]] - df / 2.,
                                       f_channels[top_lim[-1] - 1] + df / 2.,
                                       9), 1)

    ax_wat_map.set_yticks(yticks)
    ax_wat_map.set_yticklabels(yticklabels, fontsize=14)

    # plot summed profile
    wat_prof = np.nansum(waterfall, axis=0)
    plot_wat_prof, = ax_wat_prof.plot(wat_prof, fg_color + '-', linewidth=2)
    ax_wat_prof.set_ylim([wat_prof.min(), wat_prof.max()])
    ax_wat_prof.set_xlim([0, wat_prof.size])
    ax_wat_prof.set_title("Waterfall", fontsize=16, color=fg_color, y=1.08)

    # plot instructions
    text = """
    Manual selection of
      frequency range.

    On the plot, press
      "t" to select top limit.
      "b" to select bottom limit.
      "T" to undo upper limit.
      "B" to undo lower limit.
      "s" to subband by factor 2.
      "S" to upband by factor 2.
      "q" to save and exit.

    """
    instructions = ax_text.annotate(text, (0, 1), color=fg_color, fontsize=14,
                                    horizontalalignment='left',
                                    verticalalignment='top', linespacing=1.5)

    def subband(data):
        """Downsample frequency channels to guide the eye."""
        nsamp = data.shape[1]
        return np.nansum(data.reshape(-1, sub_factor[-1], nsamp), axis=1)

    # GUI for observing frequency selection
    def update_lim():
        """Replot figure with newly selected bottom or top limit."""
        plot_wat_map.set_data(
            subband(waterfall)[bottom_lim[-1]:top_lim[-1], ...])
        plot_wat_map.autoscale()
        plot_wat_map.set_extent((left, right, bottom_lim[-1], top_lim[-1] + 1))

        # set frequencies as label instead of channel numbers
        yticks = np.linspace(bottom_lim[-1], top_lim[-1] + 1, 9)
        sub_f_channels = f_channels.reshape(-1, sub_factor[-1]).mean(axis=1)
        yticklabels = np.round(np.linspace(
            sub_f_channels[bottom_lim[-1]] - df * sub_factor[-1] / 2.,
            sub_f_channels[top_lim[-1] - 1] + df * sub_factor[-1] / 2., 9), 1)

        ax_wat_map.set_yticks(yticks)
        ax_wat_map.set_yticklabels(yticklabels, fontsize=14)

        wat_prof = np.nansum(subband(
            waterfall)[bottom_lim[-1]:top_lim[-1], ...], axis=0)
        plot_wat_prof.set_ydata(wat_prof)
        ax_wat_prof.set_ylim([wat_prof.min(), wat_prof.max()])
        return

    def press(event):
        """Read user input."""
        sys.stdout.flush()
        if event.key == "t":
            y = int(round(event.ydata))
            top_lim.append(y)
            update_lim()
        if event.key == "b":
            y = int(round(event.ydata))
            bottom_lim.append(y)
        elif event.key == "s":
            # subband factor should leave at least 8 channels
            if sub_factor[-1] * 8 != top_lim[0] * sub_factor[-1]:
                sub_factor.append(sub_factor[-1] * 2)
                top_lim[:] = [x // 2 for x in top_lim]
                bottom_lim[:] = [x // 2 for x in bottom_lim]
                update_lim()
        elif event.key == "T":
            if len(top_lim) > 1:
                del top_lim[-1]
            update_lim()
        elif event.key == "B":
            if len(bottom_lim) > 1:
                del bottom_lim[-1]
            update_lim()
        elif event.key == "S":
            if len(sub_factor) > 1:
                del sub_factor[-1]
                top_lim[:] = [x * 2 for x in top_lim]
                bottom_lim[:] = [x * 2 for x in bottom_lim]
                update_lim()
        fig.canvas.draw()
        return

    def new_cmap(event):
        """Change colormap."""
        COLORMAP = next(COLORMAP_LIST)
        plot_wat_map.set_cmap(COLORMAP)
        plot_pow_map.set_cmap(COLORMAP)
        fig.canvas.draw()
        return

    ax_but = plt.axes([0.01, 0.94, 0.18, 0.05])
    but = Button(ax_but, 'Change colormap', color='0.8', hovercolor='0.2')
    but.on_clicked(new_cmap)

    try:
        cursor = Cursor(ax_wat_map, color='g', linewidth=2, vertOn=False)
    except AttributeError:
        pass

    key = fig.canvas.mpl_connect('key_press_event', press)
    plt.show()

    return bottom_lim[-1] * sub_factor[-1], top_lim[-1] * sub_factor[-1]


def _get_f_threshold_manual(power_spectra, dpower_spectra, waterfall, dm_list,
                            f_channels, t_res, ref_freq="top"):
    """Select power limits with interactive GUI.

    Parameters
    ----------
    power_spectra : array_like
        'Coherence' power spectra for a range of trial DMs.
    dpower_spectra : array_like
        Derivative 'coherence' power spectra for a range of trial DMs.
    waterfall : array_like
        Burst dynamic spectrum.
    dm_list : array_like
        List of dispersion measure values to search, in pc/cc.
    f_channels : array_like
        Center frequencies, in MHz.
    t_res : float
        Sampling time, in s.
    ref_freq : str, optional
        Reference frequency for dedispersion, one of
        ["top", "center", "bottom"]. "Top" by default.

    Returns
    -------
    tuple of int
        Bottom and top fluctuation frequency index and phase limit to
        use.

    """
    bg_color = "k"
    fg_color = "w"

    # define axes
    fig = plt.figure(figsize=(12., 8.5), facecolor='k')
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.95, top=0.94, hspace=0)
    gs = gridspec.GridSpec(2, 3, hspace=0, wspace=0.02, height_ratios=[1, 4],
                           width_ratios=[2, 3, 2])
    ax_text = fig.add_subplot(gs[1, 0])
    ax_pow_prof = fig.add_subplot(gs[0, 1])
    ax_pow_map = fig.add_subplot(gs[1, 1], sharex=ax_pow_prof)
    ax_wat_prof = fig.add_subplot(gs[0, 2])
    ax_wat_map = fig.add_subplot(gs[1, 2], sharex=ax_wat_prof)

    for ax in fig.axes:
        ax.axis('off')

    # plot power
    plot_pow_map = ax_pow_map.imshow(power_spectra, origin='lower',
                                     aspect='auto', cmap=COLORMAP,
                                     interpolation='none')
    ax_pow_map.set_ylim([0, power_spectra.shape[0]])
    pow_prof = dpower_spectra.sum(axis=0)
    plot_pow_prof, = ax_pow_prof.plot(pow_prof, 'w-', linewidth=2,
                                      clip_on=False)
    ax_pow_prof.set_ylim([pow_prof.min(), pow_prof.max()])
    ax_pow_prof.set_title('Coherent power', fontsize=16, color='w',
                          y=1.08)

    # plot waterfall
    top_lim = [power_spectra.shape[0],]
    bottom_lim = [1,]
    dm, _ = _dm_calculation(waterfall, power_spectra, dpower_spectra,
                            bottom_lim[-1], top_lim[-1], f_channels, t_res,
                            dm_list, no_plots=True, fname="", phase_lim=None,
                           blackonwhite=False, fformat=".pdf", dm_curve=None, weight=None)
    waterfall_dedisp = _dedisperse_waterfall(waterfall, dm, f_channels, t_res,
                                             ref_freq=ref_freq)
    plot_wat_map = ax_wat_map.imshow(waterfall_dedisp, origin='lower',
                                     aspect='auto', cmap=COLORMAP,
                                     interpolation='none')
    wat_prof = waterfall_dedisp.sum(axis=0)
    plot_wat_prof, = ax_wat_prof.plot(wat_prof, 'w-', linewidth=2)
    ax_wat_prof.set_ylim([wat_prof.min(), wat_prof.max()])
    ax_wat_prof.set_xlim([0, wat_prof.size])
    ax_wat_prof.set_title("Waterfall", fontsize=16, color='w', y=1.08)

    # plot instructions
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
    instructions = ax_text.annotate(text.format(dm), (0, 1), color='w',
                                    fontsize=14, horizontalalignment='left',
                                    verticalalignment='top', linespacing=1.5)

    # GUI for fluctuation frequency selection
    def update_lim(is_log):
        """Replot figure with newly selected bottom or top limit."""
        if is_log:
            pow_map = np.log10(power_spectra[bottom_lim[-1]:top_lim[-1]])
        else:
            pow_map = power_spectra[bottom_lim[-1]:top_lim[-1]]

        plot_pow_map.set_clim(vmin=pow_map.min(), vmax=pow_map.max())
        ax_pow_map.set_ylim([bottom_lim[-1], top_lim[-1]])
        pow_prof = dpower_spectra[bottom_lim[-1]:top_lim[-1]].sum(axis=0)
        plot_pow_prof.set_ydata(pow_prof)
        ax_pow_prof.set_ylim([pow_prof.min(), pow_prof.max()])
        dm, _ = _dm_calculation(waterfall, power_spectra, dpower_spectra,
                                bottom_lim[-1], top_lim[-1], f_channels, t_res,
                                dm_list, no_plots=True, fname="",phase_lim=None,
                                blackonwhite=False, fformat=".pdf", dm_curve=None, weight=None)
        waterfall_dedisp = _dedisperse_waterfall(waterfall, dm, f_channels,
                                                 t_res, ref_freq=ref_freq)
        plot_wat_map.set_data(waterfall_dedisp)
        wat_prof = waterfall_dedisp.sum(axis=0)
        plot_wat_prof.set_ydata(wat_prof)
        ax_wat_prof.set_ylim([wat_prof.min(), wat_prof.max()])
        instructions.set_text(text.format(dm))
        return

    is_log = [False]
    def press(event):
        """Read user input."""
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
                plot_pow_map.set_data(power_spectra)
                plot_pow_map.set_clim(
                    vmin=power_spectra[bottom_lim[-1]:top_lim[-1]].min(),
                    vmax=power_spectra[bottom_lim[-1]:top_lim[-1]].max())
                is_log[0] = False
            else:
                power_spectra_log = np.log10(power_spectra)
                plot_pow_map.set_data(power_spectra_log)
                plot_pow_map.set_clim(
                    vmin=power_spectra_log[bottom_lim[-1]:top_lim[-1]].min(),
                    vmax=power_spectra_log[bottom_lim[-1]:top_lim[-1]].max())
                is_log[0] = True
        elif event.key == " ":
            ax_wat_prof.set_xlim([0, wat_prof.size])
            xlim[0] = 0
            xlim[1] = wat_prof.size
        fig.canvas.draw()
        return

    xlim = [0, wat_prof.size]
    def onselect_prof(xmin, xmax):
        """Select phase window in the burst profile."""
        ax_wat_prof.set_xlim(xmin, xmax)
        xlim[0] = int(xmin)
        xlim[1] = int(xmax)
        fig.canvas.draw()
        return

    def onselect_map(xmin, xmax):
        """Select phase window in the waterfall."""
        ax_wat_prof.set_xlim(xmin, xmax)
        xlim[0] = int(xmin)
        xlim[1] = int(xmax)
        fig.canvas.draw()
        return

    def new_cmap(event):
        """Change colormap."""
        COLORMAP = next(COLORMAP_LIST)
        plot_wat_map.set_cmap(COLORMAP)
        plot_pow_map.set_cmap(COLORMAP)
        fig.canvas.draw()
        return

    ax_but = plt.axes([0.01, 0.94, 0.12, 0.05])
    but = Button(ax_but, 'Change colormap', color='0.8', hovercolor='0.2')
    but.on_clicked(new_cmap)

    span_prof = SpanSelector(ax_wat_prof, onselect_prof, 'horizontal',
                             rectprops=dict(alpha=0.5, facecolor='g'))
    span_map = SpanSelector(ax_wat_map, onselect_map, 'horizontal',
                            rectprops=dict(alpha=0.5, facecolor='g'))

    try:
        cursor = Cursor(ax_pow_map, color='g', linewidth=2, vertOn=False)
    except AttributeError:
        pass

    fig.canvas.mpl_connect('key_press_event', press)

    plt.show()

    return bottom_lim[-1], top_lim[-1], xlim


def _poly_max(x, y, err, w='None'):
    """
    Polynomial fit
    """
    ## AS: matrix_rank was hanging on large arrays
    if np.shape(x)[0] < 7:
        n = np.linalg.matrix_rank(np.vander(y))
    else:
        n=6
    dx = x - x.mean()

    if w is None:
      p = np.polyfit(dx, y, n)
      err = max( [ np.std(y-np.polyval(p, dx)),  err] )
    else:
      p = np.polyfit(dx,y,n,w = w)
      err = max([ (np.sum( np.multiply(w , (y-np.polyval(p, dx))**2.0 ) )/np.sum(w))**0.5,  err])
      #err = max( [ np.std(y-np.polyval(p, dx)),  err] )
    dp = np.polyder(p)
    ddp = np.polyder(dp)
    cands = np.roots(dp)
    r_cands = np.polyval(ddp, cands)
    first_cut = cands[(cands.imag==0) & (cands.real>=min(dx)) & (cands.real<=max(dx)) & (r_cands<0)]
    if first_cut.size > 0:
        value = np.polyval(p, first_cut)
        best = np.real(first_cut[value.argmax()])
        delta_x = np.sqrt(np.abs( 2.0 * err / np.polyval(ddp, best)) )
    else:
        best = 0.
        delta_x = 0.

    return float( np.real(best) + x.mean() ), delta_x, p, x.mean()


def _plot_power(dm_map, low_idx, up_idx, X, Y, plot_range, returns_poly, x, y,
                snr, t_res, fname="", blackonwhite=False, fformat=".pdf"):
    """Diagnostic plot of coherent power vs dispersion measure."""
    if low_idx==0:
        low_idx=1
    if blackonwhite:
        bg_color = "w"
        fg_color = "k"
    else:
        bg_color = "k"
        fg_color = "w"
    if fformat not in [".pdf", ".png"]:
        fformat = ".pdf"

    fig = plt.figure(figsize=(6, 8.5), facecolor=bg_color)
    fig.subplots_adjust(left=0.1, bottom=0.05, right=0.99, top=0.88)
    gs = gridspec.GridSpec(3, 1, hspace=0, height_ratios=[3, 1, 9])
    ax_prof = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_prof)
    ax_map = fig.add_subplot(gs[2], sharex=ax_prof)

    title = "{0:}\nBest DM = {1:.3f} $\pm$ {2:.3f}\nS/N = {3:.1f}".format(
        fname, returns_poly[0], returns_poly[1], snr)
    fig.suptitle(title, color=fg_color, linespacing=1.5)

    # Profile
    ax_prof.plot(X, Y, fg_color+'-', linewidth=3, clip_on=False)
    ax_prof.plot(
        X[plot_range],
        np.polyval(returns_poly[2], X[plot_range] - returns_poly[3]),
        color='orange',
        linewidth=3,
        zorder=2,
        clip_on=True
    )
    ax_prof.set_xlim([X.min(), X.max()])
    ax_prof.set_ylim([Y.min(), Y.max()])
    ax_prof.axis('off')
    #ax_prof.set_ylabel("SNR")
    ax_prof.tick_params(axis='both', colors=fg_color, labelbottom=False,
                       labelleft=False, direction='in', left=False, top=True)
    ax_prof.yaxis.label.set_color(fg_color)
    try:
        ax_prof.set_facecolor(bg_color)
    except AttributeError:
        ax_prof.set_axis_bgcolor(bg_color)
    ax_prof.ticklabel_format(useOffset=False)

    # residuals
    residuals = y - np.polyval(returns_poly[2], x - returns_poly[3])
    residuals -= residuals.min()
    residuals /= residuals.max()
    ax_res.plot(x, residuals, 'x' + fg_color, linewidth=2, clip_on=False)
    ax_res.set_ylim([np.min(residuals) - np.std(residuals) / 2,
                     np.max(residuals) + np.std(residuals) / 2])
    ax_res.set_ylabel("$\\Delta$")
    ax_res.tick_params(axis='both', colors=fg_color, labelbottom=False,
                       labelleft=False, direction='in', left=False, top=True)
    ax_res.yaxis.label.set_color(fg_color)
    try:
        ax_res.set_facecolor(bg_color)
    except AttributeError:
        ax_res.set_axis_bgcolor(bg_color)
    ax_res.ticklabel_format(useOffset=False)

    # power vs DM map
    ft_len = dm_map.shape[0]
    idx2ang = 1. / (2 * ft_len * t_res * 1000)
    extent = [np.min(X), np.max(X), low_idx * idx2ang, up_idx * idx2ang]
    ax_map.imshow(dm_map[low_idx : up_idx], origin='lower', aspect='auto',
                  cmap=COLORMAP, extent=extent, interpolation='none')
    ax_map.tick_params(axis='both', colors=fg_color, direction='in',
                       right=True, top=True)
    ax_map.xaxis.label.set_color(fg_color)
    ax_map.yaxis.label.set_color(fg_color)
    ax_map.set_xlabel("DM (pc cm$^{-3}$)")
    # from p. 142 in pulsar handbook, also see Camilo et al. (1996)
    ax_map.set_ylabel("Fluctuation Frequency (ms$^{-1}$)")
    ax_map.ticklabel_format(useOffset=False)

    try:
        fig.align_ylabels([ax_map, ax_res])
    except AttributeError:
        ax_map.yaxis.set_label_coords(-0.07, 0.5)
        ax_res.yaxis.set_label_coords(-0.07, 0.5)

    if fname != "": fname += "_"

    fig.savefig(fname + "DM_Search" + fformat, facecolor=bg_color,
                edgecolor=bg_color, bbox_inches="tight")


def _get_window(profile):
    """ACF Windowing."""
    smooth_profile = scipy.signal.detrend(profile)
    autocorrelation = np.correlate(smooth_profile, smooth_profile, "same")
    window = np.max(np.diff(np.where(autocorrelation < 0)))

    return window


def _check_window(profile, window):
    """Check whether the viewing window will be in the index range."""
    convolved = np.convolve(1.0*profile, 1.0*np.ones(int(window)), 'same')
    peak_value = np.mean(np.where(convolved == max(convolved)))
    peak = np.where(profile == np.max(profile))

    if (peak_value - peak) ** 2 > window ** 2:
        window += np.abs(peak_value - peak) / 2
        peak_value = (peak_value + peak) / 2

    start = int(peak_value - np.round(1.25 * window))
    end = int(peak_value + np.round(1.25 * window))

    if start < 0:
        start = 0
    if end > profile.size - 1:
        end = profile.size - 1

    return start, end


def _plot_waterfall(returns_poly, waterfall, dt, f, cutoff, fname="",
                    window=None, blackonwhite=False, fformat=".pdf"):
    """Plot the waterfall at the best Dispersion Measure and at close
    values for comparison.

    """
    if blackonwhite:
        bg_color = "w"
        fg_color = "k"
    else:
        bg_color = "k"
        fg_color = "w"
    if fformat not in [".pdf", ".png"]:
        fformat = ".pdf"

    fig = plt.figure(figsize=(8.5, 6), facecolor=bg_color)
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.99, top=0.8)
    grid = gridspec.GridSpec(1, 3, wspace=0.1)

    title = "{0:}\nBest DM = {1:.3f} $\\pm$ {2:.3f}".format(
        fname, returns_poly[0], returns_poly[1])
    plt.suptitle(title, color=fg_color, linespacing=1.5)

    # DMs +/- 5 sigmas away
    dms = returns_poly[0] + 5 * returns_poly[1] * np.array([-1, 0, 1])
    for j, dm in enumerate(dms):
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[j],
                                              height_ratios=[1, 4], hspace=0)
        ax_prof = fig.add_subplot(gs[0])
        ax_wfall = fig.add_subplot(gs[1], sharex=ax_prof)

        try:
            ax_wfall.set_facecolor(bg_color)
        except AttributeError:
            ax_wfall.set_axis_bgcolor(bg_color)

        wfall = _dedisperse_waterfall(waterfall, dm, f, dt)
        profile = wfall.sum(axis=0)

        # find the time range around the pulse
        if (j == 0) and (window is None):
          #wfall1 = _dedisperse_waterfall(waterfall, dms[1], f, dt)
          width = np.max([ _get_window(profile), np.shape(profile)[0]/cutoff ])
          #coherence_spectrum = _get_coherence_spectrum(wfall1)
          #spectrum_filter = np.ones_like(coherence_spectrum)
          #spectrum_filter[cutoff:-cutoff] = 0
          #spike = np.real(ifft(np.multiply(fft(profile),np.multiply(coherence_spectrum, spectrum_filter))))
          #spike[0] = 0
          #spike[-1] = 0

        kern_l = np.shape(profile)[0]/cutoff
        spike  = np.convolve( profile, np.ones(int(kern_l)), mode='same')
        window = _check_window(spike, 2 * width)

        tmax = dt * (window[1] - window[0]) * 1000
        x = np.linspace(0, tmax, window[1] - window[0])
        y = profile[window[0]:window[1]]
        ax_prof.plot(x, y, fg_color, linewidth=0.5, clip_on=False)
        ax_prof.axis('off')
        ax_prof.set_title('{0:.3f}'.format(dm), color=fg_color)

        # waterfall
        im = wfall[:, window[0]:window[1]]
        extent = [0, tmax, f[0], f[-1]]
        vmin = wfall.mean() - wfall.std()
        vmax = wfall.max()
        ax_wfall.imshow(im, origin='lower', aspect='auto', cmap=COLORMAP,
                        extent=extent, interpolation='none', vmin=vmin,
                        vmax=vmax)

        ax_wfall.tick_params(axis='both', colors=fg_color, direction='in',
                             right=True, top=True)
        if j == 0:
            ax_wfall.set_ylabel('Frequency (MHz)')
        if j == 1:
            ax_wfall.set_xlabel('Time (ms)')
        if j > 0:
            ax_wfall.tick_params(axis='both', labelleft=False)

        ax_wfall.yaxis.label.set_color(fg_color)
        ax_wfall.xaxis.label.set_color(fg_color)

    if fname != "":
        fname += "_"

    fig.savefig(fname + "Waterfall_5sig" + fformat, facecolor=bg_color,
                edgecolor=bg_color, bbox_inches="tight")


def _dedisperse_waterfall(wfall, dm, freq, dt, ref_freq="top"):
    """Dedisperse a waterfallfall matrix to given DM."""
    k_dm = 1. / 2.41e-4
    dedisp = np.zeros_like(wfall)

    # pick reference frequency for dedispersion
    if ref_freq == "top":
        reference_frequency = freq[-1]
    elif ref_freq == "center":
        center_idx = len(freq) // 2
        reference_frequency = freq[center_idx]
    elif ref_freq == "bottom":
        reference_frequency = freq[0]
    else:
        print("`ref_freq` not recognized, using 'top'")
        reference_frequency = freq[-1]

    shift = (k_dm * dm * (reference_frequency**-2 - freq**-2) \
        / dt).round().astype(int)
    for i, ts in enumerate(wfall):
        dedisp[i] = np.roll(ts, shift[i])

    return dedisp


def _init_dm(fname, dm_s, dm_e):
    """Initialize DM limits of the search if not specified."""
    archive = psrchive.Archive_load(fname)
    dm = archive.get_dispersion_measure()
    if dm_s is None:
        dm_s = dm - 10
    if dm_e is None:
        dm_e = dm + 10

    return dm_s, dm_e


def from_PSRCHIVE(fname, dm_s, dm_e, dm_step, ref_freq="top",
                  manual_cutoff=False, manual_bandwidth=False, no_plots=False):
    """Brute-force search of the dispersion measure of a single pulse
    stored into a PSRCHIVE file. The algorithm uses phase information
    and is robust to interference and unusual burst shapes.

    Parameters
    ----------
    fname : str
        Name of a PSRCHIVE file.
    dm_s : float
        Starting value of dispersion measure to search, in pc/cc.
    dm_e : float
        Ending value of dispersion measure to search, in pc/cc.
    dm_step : float
        Step of the search, in pc/cc.

    Returns
    -------
    dm : float
        Best value of Dispersion Measure (pc/cc).
    dm_std :
        Standard deviation of the Dispersion Measure (pc/cc)

    Stores
    ------
    basename(fname) + "_Waterfall_5sig.pdf" : plot
        Pulse waterfall at the best Dispersion Measure and 5 sigmas away
    basename(fname) + "_DM_Search.pdf": plot
        Map of the coherent power as a function of the search Dispersion
        Measure.

    """
    waterfall, f_channels, t_res = _load_psrchive(fname)
    dm_s, dm_e = _init_dm(fname, dm_s, dm_e)
    dm_list = np.arange(np.float(dm_s), np.float(dm_e), np.float(dm_step))
    dm, dm_std = get_dm(waterfall, dm_list, t_res, f_channels,
                        ref_freq=ref_freq, manual_cutoff=manual_cutoff,
                        manual_bandwidth=manual_bandwidth,
                        fname=os.path.basename(fname), no_plots=no_plots)

    return dm, dm_std


def get_dm(waterfall, dm_list, t_res, f_channels, ref_freq="top",
           manual_cutoff=False, manual_bandwidth=False, fname="",
           no_plots=False, blackonwhite=False, fformat=".pdf",
           output_snr=False):
    """Brute-force search of the Dispersion Measure of a waterfall numpy
    matrix. The algorithm uses phase information and is robust to
    interference and unusual burst shapes.

    Parameters
    ----------
    waterfall : ndarray
        2D array with shape (frequency channels, phase bins)
    dm_list : list
        List of Dispersion Measure values to search (pc/cc).
    t_res : float
        Time resolution of each phase bin (s).
    f_channels : list
        Central frequency of each channel, from low to high (MHz).
    ref_freq : str, optional. Default = "top"
        Use either the "top", "center" or "bottom" of the band as
        reference frequency for dedispersion.
    manual_cutoff : bool, optional. Default = False
        If False, the power spectrum cutoff is automatically selected.
    manual_bandwidth : bool, optional. Default = False
        If False, use the full frequency bandwidth.
    fname : str, optional. Default = ""
        Filename used as a prefix for the diagnostic plots.
    output_snr : bool, optional. Default = False
        If True, return the peak S/N as a third variable.

    Returns
    -------
    dm : float
        Best value of dispersion measure, in pc/cc.
    dm_std :
        Standard deviation of the dispersion measure, in pc/cc.
    output_snr : float
        Peak S/N, only when `output_snr` is True.

    """
    if manual_bandwidth:
        low_ch_idx, up_ch_idx = _get_frequency_range_manual(waterfall,
                                                            f_channels)
    else:
        low_ch_idx = 0
        up_ch_idx = waterfall.shape[0]

    waterfall = waterfall[low_ch_idx:up_ch_idx, ...]
    f_channels = f_channels[low_ch_idx:up_ch_idx]

    nchan = waterfall.shape[0]
    nbin = waterfall.shape[1] // 2
    power_spectra = np.zeros([nbin, dm_list.size])

    for i, dm in enumerate(dm_list):
        waterfall_dedisp = _dedisperse_waterfall(
            waterfall,
            dm,
            f_channels,
            t_res,
            ref_freq=ref_freq
        )
        power_spectrum = _get_coherent_power_spectrum(waterfall_dedisp)
        power_spectra[:, i] = power_spectrum[: nbin]

    # weight by fluctuation frequency index
    ff_idx = np.arange(0, nbin)
    dpower_spectra = power_spectra * ff_idx[:, np.newaxis] ** 2

    # based on Gamma(2,)
    mean = nchan
    std = nchan / np.sqrt(2)

    if manual_cutoff:
        low_idx, up_idx, phase_lim = _get_f_threshold_manual(
            power_spectra,
            dpower_spectra,
            waterfall,
            dm_list,
            f_channels,
            t_res,
            ref_freq=ref_freq
        )
        dm_curve = None
        w = None
        dstd = None
        SN = None
    else:
        low_idx, up_idx = _get_f_threshold(power_spectra, mean, std)
        phase_lim = None
        dm_curve , dm_c_err, SNR = _get_dm_curve(power_spectra, dpower_spectra, nchan)
        dm_curve[SNR<5.0]=dm_curve[SNR<5.0]/1e6
        #w = None
        w = SNR
        w[np.isnan(w)] = 0.0
        w[SNR<5.0]     = 1/1e6  # Setting to Zero Mess with low SNR cands
        #w  = np.exp(w)
        w = w / np.sum(w)
        dstd = np.max(dm_c_err)
        #w = np.divide(dm_curve,SN_Err)
        #w[np.isnan(w)]=0.
        #w = (w )/np.sum(w)


    dm, dm_std = _dm_calculation(
        waterfall,
        power_spectra,
        dpower_spectra,
        low_idx,
        up_idx,
        f_channels,
        t_res,
        dm_list,
        no_plots = no_plots,
        fname = fname,
        fformat = fformat,
        phase_lim = phase_lim,
        blackonwhite = blackonwhite,
        dm_curve = dm_curve,
        weight   = w,
        dstd     = dstd,
        SN = np.max(SNR)
    )
    if output_snr:
        return dm, dm_std, np.max(SNR)
    else:
        return dm, dm_std


def _dm_calculation(waterfall, power_spectra, dpower_spectra, low_idx, up_idx,
                    f_channels, t_res, dm_list, no_plots=False, fname="",
                    phase_lim=None, blackonwhite=False, fformat=".pdf",
                    dm_curve=None,weight=None,dstd=None, SN = None):
    """Calculate the best DM value."""
    fact_idx = up_idx - low_idx
    nchan = len(f_channels)
    if dm_curve is None:
        dm_curve = dpower_spectra[low_idx:up_idx].sum(axis=0)


        max_dm = dm_curve.max()


        # based on Gamma(2,)
        mean = 2.0*nchan
        std = mean / np.sqrt(2)

        m_fact = np.sum(np.arange(low_idx, up_idx) ** 2)
        s_fact = np.sum(np.arange(low_idx, up_idx) ** 4) ** 0.5
        dmean = mean * m_fact
        I = np.transpose( 1.0*np.ones([dm_list.size, 1]) * ( 1.0*np.arange(low_idx,up_idx) ) ** 2.0 )
        dstd1 = ( 1.0*(std * s_fact)**2.0 + np.sum( np.multiply(I, power_spectra[low_idx:up_idx]**2.0) ,axis=0) )**0.5
        dm_curve = np.divide( (dm_curve -dmean) , dstd1 )
        weight = np.multiply(dm_curve, dstd1**-1.0)
        dstd = np.max(dstd1)

    #else:
         #dstd = 2**0.5       #Overwrite terms to get correct values from _get_dm_curve (min error possible)

    #snr = SNR_curve[np
    max_dm = np.max(dm_curve)
    #snr = (max_dm - 2*nchan) / (nchan*6.**0.5)
    #snr = (max_dm - dmean) / dstd
    snr = SN


    if weight is None:
      peak = dm_curve.argmax()
      width = _get_window(dm_curve) / 2
      Start,Stop = _check_window(dm_curve, width)
    else:
      w_dm_curve = np.multiply(weight,dm_curve)
      peak = dm_curve.argmax()
      curve = power_spectra[low_idx+1:low_idx+2].sum(axis=0)
      width = int(_get_window(curve) / 4)
      #width = np.size(dm_curve)
      #Start,Stop = _check_window(w_dm_curve, width)
      #Heavy_weights = np.argwhere(weight>1e-6)
      Heavy_weights = np.argwhere( dm_curve > .5*dm_curve[peak] )
      if len(Heavy_weights) < 5:
        Heavy_weights = np.argwhere( dm_curve > .25*dm_curve[peak] )
      if len(Heavy_weights) < 5:
        Heavy_weights = np.argwhere( dm_curve > .1*dm_curve[peak] )
      #width = int(np.mean((Heavy_weights-peak)**2)**0.5/2)
      peak  = np.mean(Heavy_weights)
      width = (np.max(Heavy_weights)-np.min(Heavy_weights))
      Start = Heavy_weights[np.argmin(np.absolute( (peak - width)-Heavy_weights ) ) ]
      Stop =  Heavy_weights[np.argmin(np.absolute( (peak + width)-Heavy_weights ) ) ]

      if Start< 0: Start=0
      if Stop > np.size(w_dm_curve): Stop = np.size(w_dm_curve)
    #plot_range = np.arange(peak - width, peak + width)
    plot_range = np.arange(Start,Stop)
    y = dm_curve[plot_range]
    x = dm_list[plot_range]


    #snr = max_dm

    if weight is None:
      New_W = 1.0 *  np.ones(x.shape) / np.sum( np.ones(x.shape) )
    else:
      New_W = weight[plot_range]/np.sum(weight[plot_range])

    returns_poly = _poly_max(x, y, dstd, w = New_W)
    #print returns_poly

    if not no_plots:
        _plot_power(power_spectra, low_idx, up_idx, dm_list, dm_curve,
                    plot_range, returns_poly, x, y, snr, t_res, fname=fname,
                    fformat=fformat, blackonwhite=blackonwhite)
        _plot_waterfall(returns_poly, waterfall, t_res, f_channels, fact_idx,
                        fname=fname, fformat=fformat, window=phase_lim,
                        blackonwhite=blackonwhite)

    dm = returns_poly[0]
    dm_std = returns_poly[1]

    return dm, dm_std


def _get_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Search for best DM based on FFT phase angles.")
    parser.add_argument('fname', help="Filename of the PSRCHIVE file.")
    parser.add_argument(
        "-DM_s",
        help="Start DM. If None, DM will be selected from the PSRCHIVE file.",
        default=None, type=float)
    parser.add_argument(
        "-DM_e",
        help="End DM. If None, DM will be selected from the PSRCHIVE file.",
        default=None, type=float)
    parser.add_argument(
        "-DM_step", help="Step DM.", default=0.1, type=float)
    parser.add_argument(
        "-ref_freq",
        help="Reference frequency for dedispersion, " \
             + "one of {'top', 'center', 'bottom'}.",
        default="top", type=str)
    parser.add_argument(
        "-manual_cutoff", help="Manually set the FFT frequency cutoff.",
        action='store_true')
    parser.add_argument(
        "-manual_bandwidth",
        help="Manually set the frequency bandwidth to use.",
        action='store_true')
    parser.add_argument(
        "-no_plots", help="Do not produce diagnostic plots.",
        action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = _get_parser()
    import psrchive
    dm, dm_std = from_PSRCHIVE(
        args.fname,
        args.DM_s,
        args.DM_e,
        args.DM_step,
        ref_freq = args.ref_freq,
        manual_cutoff = args.manual_cutoff,
        manual_bandwidth = args.manual_bandwidth,
        no_plots=args.no_plots
    )
