import numpy as np
import matplotlib.pyplot as plt


def ex1():
    pass


def ex2():
    pass


# def ex3():
#     A = 3
#     N = 10
#     signal = [A * (1 - (n % N) / N) for n in range(N)]

#     values = [0, N, 4*N, 9*N]

#     for value in values:
#         with_zeros = np.zeros(N + value)
#         with_zeros[:N] = signal

#         amplitude_spectrum = np.abs(np.fft.fft(signal))
#         plt.stem(amplitude_spectrum)
#         plt.show()

#         phase_spectrum = np.angle(np.fft.fft(signal))
#         plt.stem(phase_spectrum)
#         plt.show()


def ex3():
    A = 3
    N = 10
    signal = np.array([A * (1 - (n % N) / N) for n in range(N)])
    values = [0, N, 4*N, 9*N]

    fig, axs = plt.subplots(len(values), 2, figsize=(12, 8))

    for i, value in enumerate(values):
        with_zeros = np.zeros(N + value)
        with_zeros[:N] = signal

        amp_spectrum = np.abs(np.fft.fft(with_zeros))
        phase_spectrum = np.angle(np.fft.fft(with_zeros))

        axs[i, 0].stem(amp_spectrum, linefmt='b', markerfmt='b', basefmt='b')
        axs[i, 0].set_title(f'Widmo Amplitudowe ({int(value / 10)}N)')
        axs[i, 0].grid(True)

        axs[i, 1].stem(phase_spectrum, linefmt='r', markerfmt='r', basefmt='r')
        axs[i, 1].set_title(f'Widmo Fazowe ({int(value / 10)}N)')
        axs[i, 1].grid(True)

    plt.tight_layout()
    plt.show()


def ex4():
    A1, f1 = 0.1, 3000
    A2, f2 = 0.4, 4000
    A3, f3 = 0.8, 10000
    fs = 48000

    N1 = 2048
    N2 = 3072

    def signal_r(A1, A2, A3, f1, f2, f3, t):
        return (A1 * np.sin(2 * np.pi * f1 * t) +
                A2 * np.sin(2 * np.pi * f2 * t) +
                A3 * np.sin(2 * np.pi * f3 * t))

    def psd(signal):
        N = len(signal)
        fft_val = np.fft.fft(signal)
        psd_values = 2 * np.abs(fft_val) / N
        psd_values[0] = psd_values[0] / 2
        return psd_values

    s1 = np.array([signal_r(A1, A2, A3, f1, f2, f3, n / fs)
                   for n in range(N1)])
    s2 = np.array([signal_r(A1, A2, A3, f1, f2, f3, n / fs)
                   for n in range(N2)])

    psd1 = psd(s1)
    freq1 = np.fft.fftfreq(N1, 1/fs)

    psd2 = psd(s2)
    freq2 = np.fft.fftfreq(N2, 1/fs)

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 2, 1)
    plt.stem(freq1, psd1)
    plt.xlim(0, 12000)
    plt.title(f'N1={N1}')

    plt.subplot(2, 2, 2)
    plt.stem(freq2, psd2)
    plt.xlim(0, 12000)
    plt.title(f'N2={N2}')

    plt.subplot(2, 2, 3)
    plt.psd(s1, Fs=fs)
    plt.xlim(0, 12000)
    plt.axvline(f1, color='r', linestyle='--', label='f1 = 3000 Hz')
    plt.axvline(f2, color='g', linestyle='--', label='f2 = 4000 Hz')
    plt.axvline(f3, color='b', linestyle='--', label='f3 = 10000 Hz')

    plt.subplot(2, 2, 4)
    plt.psd(s2, Fs=fs)
    plt.xlim(0, 12000)
    plt.axvline(f1, color='r', linestyle='--', label='f1 = 3000 Hz')
    plt.axvline(f2, color='g', linestyle='--', label='f2 = 4000 Hz')
    plt.axvline(f3, color='b', linestyle='--', label='f3 = 10000 Hz')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ex1()
    # ex2()
    ex3()
    # ex4()
