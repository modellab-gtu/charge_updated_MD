import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse

# ---------------- CONFIGURATION ----------------
#FILE_PATH = "nvtmd_updates.dat"  # Your data file
# -----------------------------------------------

def plot_diagnostics(file_path):
    try:
        # Load data. 
        # The file has no header in the snippet, but the previous code writes one.
        # We assume the columns based on the code provided previously:
        # 0:seg, 1:step, 2:time, 3:dPE, 4:shift, 5:mean, 6:std, 7:min, 8:max, 9:qsum
        cols = ["seg", "step", "time_ps", "dPE", "shift", "dq_mean", "dq_std", "dq_min", "dq_max", "q_total"]
        
        # Try reading with header first, if that fails (or looks wrong), read without
        try:
            df = pd.read_csv(file_path, sep="\t")
            if "dPE" not in df.columns: raise ValueError
        except:
            df = pd.read_csv(file_path, sep="\t", names=cols, header=None)

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot 1: Energy Shock (dPE)
        ax = axes[0]
        ax.plot(df["time_ps"], df["dPE"], 'r-o', label="Delta PE (kJ/mol)")
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel("$\Delta$ Potential Energy (kJ/mol)")
        ax.set_title("Energy Shock per Update (Ideally approaches 0)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Charge Drift Stats
        ax = axes[1]
        ax.plot(df["time_ps"], df["dq_max"], 'g-', label="Max $\Delta q$ (+)", alpha=0.7)
        ax.plot(df["time_ps"], df["dq_min"], 'b-', label="Min $\Delta q$ (-)", alpha=0.7)
        ax.fill_between(df["time_ps"], df["dq_min"], df["dq_max"], color='gray', alpha=0.1)
        ax.axhline(0.05, color='red', linestyle=':', label="Stability Limit (0.05)")
        ax.axhline(-0.05, color='red', linestyle=':')
        ax.set_ylabel("Charge Change ($\Delta e$)")
        ax.set_title("Magnitude of Charge Updates")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Plot 3: Net Charge Conservation
        ax = axes[2]
        ax.plot(df["time_ps"], df["q_total"], 'k.-', label="Total Charge")
        ax.set_ylabel("Total System Charge (e)")
        ax.set_xlabel("Simulation Time (ps)")
        ax.set_title("Total Charge Conservation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Zoom in on y-axis for charge to see tiny drifts
        mean_q = df["q_total"].mean()
        ax.set_ylim(mean_q - 0.01, mean_q + 0.01)

        plt.tight_layout()
        plt.savefig("charge_diagnostics.png", dpi=150)
        print("Plot saved to charge_diagnostics.png")
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot molecular dynamics energetics data from two files.")

    # Add the --upd argument for the updates file path
    parser.add_argument(
        '--upd',
        type=str,
        required=True,
        help="Path to the updates data file (e.g., nvtmd_updates.dat)."
    )

    args = parser.parse_args()

    # Call the plotting function with the paths provided via command-line
    plot_diagnostics(args.upd)






if __name__ == "__main__":
    plot_diagnostics(FILE_PATH)
