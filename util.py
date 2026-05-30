from email.mime import audio

import numpy as np
from matplotlib import pyplot as plt
from pydub import AudioSegment
from scipy import signal
from scipy.signal import find_peaks


def read_signal_from_mp3(input_file):
    sound = AudioSegment.from_mp3(input_file)
    samples = np.frombuffer(sound.raw_data, dtype=np.int16).astype(np.float32) / 32767
    return samples, sound.frame_rate


def plot_signal(samples, sampling_rate=44100, label="Signal"):
    plt.plot(np.arange(samples.shape[0]) / sampling_rate, samples)
    plt.title(label)
    plt.xlabel("Čas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.show()


def plot_fft(y, Fs):
    N = len(y)
    x = np.linspace(0, Fs / 2, N // 2)
    ampl = 2 * abs(y[:N // 2])

    plt.plot(x, ampl / 2, label="FFT")
    plt.title('Frekvenčna vsebina')
    plt.xlabel('Frekvenca [Hz]')
    plt.ylabel('Amplituda')
    plt.legend()
    plt.grid(True)
    plt.show()


def find_harmonics(y, Fs, height=None, prominence=None, distance=None):
    N = len(y)
    x = np.linspace(0, Fs / 2, N // 2)
    ampl = 2 * abs(y[:N // 2])

    peaks, props = find_peaks(ampl, height=height, prominence=prominence, distance=distance)
    print(peaks)

    return np.sort(x[peaks])


def design_fir(y, harmonics, Fs, fo, notch_width=5):

    fc = np.array([0])
    am = np.array([1])

    for f in harmonics:
        fc = np.concatenate((fc, [(f - notch_width) / (Fs / 2), (f - notch_width) / (Fs / 2),
                                  (f + notch_width) / (Fs / 2), (f + notch_width) / (Fs / 2)]))
        am = np.concatenate((am, [1, 0, 0, 1]))

    fc = np.concatenate((fc, [1]))
    am = np.concatenate((am, [1]))

    # Izris
    plt.plot(fc * (Fs / 2), am)
    plt.title(f'Načrt filtra')
    plt.xlabel('Frekvenca [Hz]')
    plt.ylabel('Magnituda')
    plt.grid(True)
    plt.show()

    # Ta funkcija poskusi načrtovati filter glede na podan željen rezultat, tako da minimizira napako.
    taps = signal.firls(fo, fc, am)

    y_fil = signal.filtfilt(taps, 1.0, y)

    plot_signal(y_fil, Fs, label="Filtriran signal")