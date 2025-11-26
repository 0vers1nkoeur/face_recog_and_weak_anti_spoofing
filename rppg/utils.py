import matplotlib.pyplot as plt
import numpy as np

def plot_signal_buffer(rppg):
    if not rppg.signal_buffer:
        print("Signal buffer vide, rien à tracer.")
        return
    t = np.arange(len(rppg.signal_buffer)) / rppg.fps
    plt.figure(figsize=(8, 3))
    plt.plot(t, rppg.signal_buffer, label="mean green", color='green')
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.title("rPPG signal buffer")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

def plot_filtered_signal(rppg):
    if rppg.last_filtered_signal is None:
        print("Filtered signal non disponible, rien à tracer.")
        return
    t = np.arange(len(rppg.last_filtered_signal)) / rppg.fps
    plt.figure(figsize=(8, 3))
    plt.plot(t, rppg.last_filtered_signal, label="filtered signal", color='orange')
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.title("rPPG filtered signal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)