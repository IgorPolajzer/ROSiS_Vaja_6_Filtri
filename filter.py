import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

from util import plot_filter_response, plot_ffr_filter, Filter, plot_iir_filter


def find_harmonics(y, Fs, height=None, prominence=None, distance=None):
    N = len(y)
    x = np.linspace(0, Fs / 2, N // 2)
    ampl = 2 * abs(y[:N // 2])

    peaks, props = find_peaks(ampl, height=height, prominence=prominence, distance=distance)

    largest_peak_idx = peaks[np.argmax(ampl[peaks])]
    filtered_peaks = peaks[peaks != largest_peak_idx]  # remove base frequency (largest peak)

    return np.sort(x[filtered_peaks])

def design_fir(y, harmonics, Fs, fo, notch_width=5, plot=False):
    fc = np.array([0])
    am = np.array([1])

    print(f"Frequencies to filter: {harmonics}")
    for f in harmonics:
        fc = np.concatenate((fc, [(f - notch_width) / (Fs / 2), (f - notch_width) / (Fs / 2),
                                  (f + notch_width) / (Fs / 2), (f + notch_width) / (Fs / 2)]))
        am = np.concatenate((am, [1, 0, 0, 1]))

    fc = np.concatenate((fc, [1]))
    am = np.concatenate((am, [1]))


    # Ta funkcija poskusi načrtovati filter glede na podan željen rezultat, tako da minimizira napako.
    taps = signal.firls(fo, fc, am)

    if plot:
        plot_ffr_filter(fc, am, Fs)
        plot_filter_response(taps, Fs, fc, am)

    return signal.filtfilt(taps, 1.0, y)


def design_iir(y, harmonics, Fs, fo, filter_type=Filter.BUTTER,
               rp=1, rs=1, notch_width=10, plot=False):
    output = y.copy()
    sos_concat = None

    for f in harmonics:
        # Vsak harmonik = en notch filter
        low = (f - notch_width) / (Fs / 2)
        high = (f + notch_width) / (Fs / 2)

        if filter_type == Filter.BUTTER:
            sos = signal.butter(fo, [low, high], btype='bandstop', output='sos')
        elif filter_type == Filter.CHEBYSHEV_I:
            sos = signal.cheby1(fo, rp, [low, high], btype='bandstop', output='sos')
        elif filter_type == Filter.CHEBYSHEV_II:
            sos = signal.cheby2(fo, rs, [low, high], btype='bandstop', output='sos')
        elif filter_type == Filter.ELIPTICAL:
            sos = signal.ellip(fo, rp, rs, [low, high], btype='bandstop', output='sos')

        # check stability
        for entry in sos:
            a = entry[3:]
            poles = np.roots(a)

            if not np.all(np.abs(poles) < 1):
                raise Exception(f"Nestabilen filter. Poli so izven enotske krožnice.")


        # Frekvenčni odziv
        w, h = signal.freqz_sos(sos)
        fr = w * (Fs / 2) / np.pi

        #if plot:
            #plt.figure()
            #plt.plot(fr, np.abs(h), 'b')
            #plt.title(f'Frekvenčni odziv ({filter_type})')
            #plt.xlabel('Frekvenca [Hz]')
            #plt.ylabel('Ojačanje')
            #plt.grid(True)

        # Apliciraj na izhod prejšnjega filtra
        output = signal.sosfiltfilt(sos, output)

        if sos_concat is None:
            sos_concat = sos
        else:
            sos_concat = np.concatenate((sos_concat, sos))

    if plot:
        plot_iir_filter(sos_concat, Fs)

    return output