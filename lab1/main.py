import numpy as np
import matplotlib.pyplot as plt


def ex1():
    pass


def ex2():
    pass


def ex3():
    pass


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





if __name__ == "__main__":
    # ex1()
    # ex2()
    # ex3()
    ex4()
