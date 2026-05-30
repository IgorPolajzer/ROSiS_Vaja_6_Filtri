from filter import *
from util import *

if __name__ == '__main__':
    file_name = "o_high_pitch.mp3"
    fo = 41

    y, Fs = read_signal_from_mp3(fr"recordings/{file_name}")
    plot_signal(y, Fs, label=file_name)

    y_fft = np.fft.fft(y)
    plot_fft(y_fft, Fs)

    harmonics = find_harmonics(y_fft, Fs, prominence=(50, None), distance=150)
    print(harmonics)

    #y_fil = design_fir(y, harmonics, Fs, fo)
    y_fil = design_iir(y, harmonics, Fs, fo, filter_type=Filter.CHEBYSHEV_I)
    y_fil_fft = np.fft.fft(y_fil)
    plot_fft(y_fil_fft, Fs)
