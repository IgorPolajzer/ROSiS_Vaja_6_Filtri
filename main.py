import numpy as np

from util import read_signal_from_mp3, plot_fft, design_fir, find_harmonics, plot_signal

if __name__ == '__main__':
    file_name = "a_high_pitch.mp3"
    y, Fs = read_signal_from_mp3(fr"recordings/{file_name}")
    plot_signal(y, Fs)

    y_fft = np.fft.fft(y)
    plot_fft(y_fft, Fs)

    harmonics= find_harmonics(y_fft, Fs, prominence=(50, None), distance=150)
    harmonics_to_remove = [1, 2, 3, 4, 5]

    harmonics = harmonics[harmonics_to_remove]

    design_fir(y, harmonics, Fs, 8001)
