from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from pydub import AudioSegment
from scipy import signal


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


def plot_ffr_filter(fc, am, Fs):
    plt.plot(fc * (Fs / 2), am)
    plt.title(f'Načrt filtra')
    plt.xlabel('Frekvenca [Hz]')
    plt.ylabel('Magnituda')
    plt.grid(True)
    plt.show()


def plot_iir_filter(sos, Fs):
    w, h = signal.freqz_sos(sos)
    fr = w * (Fs / 2) / np.pi

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.plot(fr, 20 * np.log10(np.abs(h) + 1e-10))
    ax1.set_title('Frekvenčni odziv')
    ax1.set_ylabel('Ojačanje [dB]')
    ax1.set_xlabel('Frekvenca [Hz]')
    ax1.grid(True)

    ax2.plot(fr, np.unwrap(np.angle(h)))
    ax2.set_title('Fazni odziv')
    ax2.set_ylabel('Faza [rad]')
    ax2.set_xlabel('Frekvenca [Hz]')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_fft(y, Fs, label="Frekvenčna vsebina"):
    N = len(y)
    x = np.linspace(0, Fs / 2, N // 2)
    ampl = 2 * abs(y[:N // 2])

    plt.plot(x, ampl / 2, label="FFT")
    plt.title(label)
    plt.xlabel('Frekvenca [Hz]')
    plt.ylabel('Amplituda')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_filter_response(taps, Fs, fc=None, am=None):
    w, h = signal.freqz(taps)
    fr = w * (Fs / 2) / np.pi

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Pretvorba v dB.
    ax1.plot(fr, 20 * np.log10(np.abs(h)), label='Frekvenčni odziv')
    ax1.set_ylabel('Ojačanje [dB]')
    ax1.set_xlabel('Frekvenca [Hz]')
    ax1.grid(True)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(fc * (Fs / 2), am, color='r')
    ax1_twin.set_ylabel('Magnituda')

    # Fazni odziv
    ax2.plot(fr, np.angle(h), label="Fazni odziv")
    ax2.set_xlabel('Frekvenca [Hz]')
    ax2.set_ylabel('Faza [rad]')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


class Filter(Enum):
    BUTTER = 0
    CHEBYSHEV_I = 1
    CHEBYSHEV_II = 2
    ELIPTICAL = 3