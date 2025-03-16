import matplotlib.pyplot as plt
import numpy as np
import time

def ex1():
    N_points = 8
    period = 2  # okres
    sampling_freq = N_points / period  # częstotliwość próbkowania
    time_vals = np.linspace(0, period, N_points, endpoint=False)
    signal = np.cos(np.pi * time_vals)  # Sygnał spróbkowany

    fft_res = np.fft.fft(signal)
    amp_spectrum = np.abs(fft_res) / N_points  # Normalizacja
    frequencies = np.fft.fftfreq(N_points, d=period / N_points)
    phases = np.angle(fft_res)
    threshold = 1e-10  # wartość progowa dla amplitudy

    for i in range(N_points):
        if amp_spectrum[i] < threshold:
            phases[i] = 0
        else:
            # Dla sygnału kosinusoidalnego faza powinna wynosić -pi lub pi dla głównych składników
            if i != 0 and i != N_points // 2:  # Pomijamy składnik stały i Nyquista
                phases[i] = np.pi if np.real(fft_res[i]) < 0 else 0

    # Sprawdzanie twierdzenia Parsevala
    parseval_check = np.sum(signal ** 2) == np.sum(np.abs(fft_res) ** 2) / N_points
    print(f"Twierdzenie Parsevala: {np.sum(signal**2)} = {np.sum(np.abs(fft_res)**2) / N_points} jest {'prawdziwe' if parseval_check else 'nieprawdziwe'} ")

    # Wykresy
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.stem(time_vals, signal, linefmt="b-", markerfmt="bo", basefmt="r-")
    cos_time_x = np.arange(0, 1.75, step=0.0001)
    cos_time_y = np.cos(np.pi * cos_time_x)
    plt.plot(cos_time_x, cos_time_y, color="green")
    plt.title("Sygnał spróbkowany")
    plt.xlabel("Czas (n)")
    plt.ylabel("Amplituda")

    # Widmo amplitudowe
    plt.subplot(1, 3, 2)
    plt.stem(time_vals * sampling_freq, amp_spectrum, linefmt="b-", markerfmt="bo", basefmt="r-")
    plt.title("Widmo amplitudowe")
    plt.xlabel("N")
    plt.ylabel("Amplituda")

    # Widmo fazowe
    plt.subplot(1, 3, 3)
    plt.stem(time_vals * sampling_freq, phases, linefmt="b-", markerfmt="bo", basefmt="r-")
    plt.title("Widmo fazowe")
    plt.xlabel("N")
    plt.ylabel("Faza")

    plt.tight_layout()
    plt.show()

    # Obliczenia czasu obliczeń
    N_sizes = [2 ** i for i in range(5, 15)]
    computation_times = []
    for size in N_sizes:
        temp_times = []
        for _ in range(1000):
            time_signal = np.linspace(0, period, size, endpoint=False)
            x = np.cos(np.pi * time_signal)
            start_time = time.process_time()
            _ = np.fft.fft(x)
            end_time = time.process_time()
            temp_times.append(end_time - start_time)
        computation_times.append(np.average(temp_times))

    # Wykres czasu wykonania
    plt.figure(figsize=(8, 6))
    plt.yscale('log', base=10)
    plt.xscale('log', base=10)
    plt.plot(N_sizes, computation_times)

    plt.title("Czas wykonania dla innych n")
    plt.show()


def ex2():
    amplitude, num_samples = 2, 48
    shifts_list = [0, num_samples/4, num_samples/2, 3*num_samples/4]
    shift_labels = ["0", "N/4", "N/2", "3N/4"]

    signal_function = lambda n: amplitude * np.cos(2 * np.pi * n / num_samples)

    sample_indices = np.arange(num_samples)
    shifted_signals = [signal_function(sample_indices - shift) for shift in shifts_list]

    fft_results = [np.fft.fft(signal) for signal in shifted_signals]
    amplitude_spectra = [np.abs(fft) / num_samples for fft in fft_results]
    phase_spectra = [np.angle(fft) for fft in fft_results]

    eps_threshold = 1e-10
    for amplitude_spectrum, phase_spectrum in zip(amplitude_spectra, phase_spectra):
        phase_spectrum[amplitude_spectrum < eps_threshold] = 0
        phase_spectrum[np.abs(phase_spectrum) < eps_threshold] = 0

    fig, axes = plt.subplots(4, 2, figsize=(10, 12))

    for i, (shifted_signal, amplitude_spectrum, phase_spectrum, shift_label) in enumerate(zip(shifted_signals, amplitude_spectra, phase_spectra, shift_labels,)):
        axes[i, 0].stem(sample_indices, amplitude_spectrum, linefmt="g-", markerfmt="o", basefmt="r-")
        axes[i, 0].set(title=f"Widmo amplitudowe {shift_label}", xlabel="N", ylabel="Amplituda")

        axes[i, 1].stem(sample_indices, phase_spectrum, linefmt="c-", markerfmt="o", basefmt="r-")
        axes[i, 1].set(title=f"Widmo fazowe {shift_label}", xlabel="N", ylabel="Faza (radiany)")
        axes[i, 1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ["-π", "-π/2", "0", "π/2", "π"])

    plt.tight_layout()
    plt.show()


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
    N2 = 3 * N1 // 2

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
    freq1 = np.fft.fftfreq(N1, 1 / fs)

    psd2 = psd(s2)
    freq2 = np.fft.fftfreq(N2, 1 / fs)

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
    ex1()
    # ex2()
    # ex3()
    # ex4()
