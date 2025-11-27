import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

# Disable Matplotlib's native save shortcut that opens a dialogs.
plt.rcParams["keymap.save"] = []

class SignalPlotter:
    def __init__(self, rppg, stop_callback=None):
        self.rppg = rppg
        self.stop_callback = stop_callback
        self._save_dir = Path("data") / "plots"

    def plot_signal_buffer(self):
        t = np.arange(len(self.rppg.signal_buffer)) / self.rppg.fps
        fig = plt.figure(num="rPPG signal (raw)", figsize=(8, 3))
        fig.clf() # clear previous plots
        ax = fig.add_subplot(111)
        ax.plot(t, self.rppg.signal_buffer, label="mean green", color='green')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("rPPG signal buffer")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        self._attach_save_shortcut(fig)
        plt.show(block=False)
        plt.pause(0.001)


    def plot_filtered_signal(self):
        t = np.arange(len(self.rppg.filtered_signal_buffer)) / self.rppg.fps
        fig = plt.figure(num="rPPG signal (filtered)", figsize=(8, 3))
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(t, self.rppg.filtered_signal_buffer, label="filtered signal", color='orange')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("rPPG filtered signal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        self._attach_save_shortcut(fig)
        plt.show(block=False)
        plt.pause(0.001)

    def plot_signals(self):
        """Display the raw and filtered signals side by side in the same figure."""
        raw = self.rppg.signal_buffer
        filt = self.rppg.filtered_signal_buffer
        has_raw = len(raw) > 0
        has_filt = len(filt) > 0
        if not has_raw and not has_filt:
            return None

        fig = plt.figure(num="rPPG signals (raw + filtered)", figsize=(12, 4))
        fig.clf()
        ax_raw = fig.add_subplot(1, 2, 1)
        ax_filt = fig.add_subplot(1, 2, 2)

        if has_raw:
            t_raw = np.arange(len(raw)) / self.rppg.fps
            ax_raw.plot(t_raw, raw, label="mean green", color="green")
            ax_raw.set_title("Raw signal")
            ax_raw.legend()
        else:
            ax_raw.text(0.5, 0.5, "No raw signal", ha="center", va="center")
        ax_raw.set_xlabel("Time (s)")
        ax_raw.set_ylabel("Amplitude")
        ax_raw.grid(True, alpha=0.3)

        if has_filt:
            t_filt = np.arange(len(filt)) / self.rppg.fps
            ax_filt.plot(t_filt, filt, label="filtered signal", color="orange")
            ax_filt.set_title("Filtered signal")
            ax_filt.legend()
        else:
            ax_filt.text(0.5, 0.5, "No filtered signal", ha="center", va="center")
        ax_filt.set_xlabel("Time (s)")
        ax_filt.set_ylabel("Amplitude")
        ax_filt.grid(True, alpha=0.3)

        fig.tight_layout()
        self._attach_save_shortcut(fig)
        plt.show(block=False)
        plt.pause(0.001)
        return fig

    def _attach_save_shortcut(self, fig):
        """Bind 's' to save the figure and 'Esc' to stop the capture."""
        if getattr(fig, "_save_shortcut_attached", False):
            return

        def _handler(event, fig=fig):
            if event.key == "s":
                self._save_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H-%M-%S')
                path = self._save_dir / f"rppg_{ts}.png"
                fig.savefig(path)
                print(f"[rPPG] Figure saved: {path}")
            elif event.key in ("escape", "esc"):
                if self.stop_callback:
                    print("[rPPG] Stop requested via 'Esc' on the figure.")
                    self.stop_callback()
                plt.close('all')

        fig.canvas.mpl_connect("key_press_event", _handler)
        fig._save_shortcut_attached = True
