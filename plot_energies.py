import pandas as pd
import matplotlib.pyplot as plt
import argparse # Import the argparse module

# --- Data Loading and Plotting Function ---
def plot_energetics(ene_file_path, upd_file_path):
    """
    Loads energy and update data from the specified files and generates a 3-panel plot.
    """
    # Load energies.dat
    try:
        energies = pd.read_csv(ene_file_path, sep='\t')
    except:
        # Try whitespace if tab separation fails
        energies = pd.read_csv(ene_file_path, delim_whitespace=True)

    # Load updates.dat
    # Define columns based on previous analysis
    cols = ["seg", "step", "time_ps", "dPE", "shift", "dq_mean", "dq_std", "dq_min", "dq_max", "qsum_total"]
    try:
        updates = pd.read_csv(upd_file_path, sep='\t', header=None)
        updates.columns = cols
    except:
        # Try whitespace if tab separation fails
        updates = pd.read_csv(upd_file_path, delim_whitespace=True, header=None)
        updates.columns = cols

    # --- Plot Generation ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # --- Plot 1: Total and Potential Energy ---
    ax = axes[0]
    ax.plot(energies['time_ps'], energies['TE_kJmol'], color='black', label='Total Energy (TE)', alpha=0.8)
    ax.plot(energies['time_ps'], energies['PE_kJmol'], color='darkblue', label='Potential Energy (PE)', alpha=0.6)
    ax.set_ylabel('Energy (kJ/mol)')
    ax.set_title('Total and Potential Energy vs. Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Temperature and Energy Injection (dPE) ---
    ## Create a figure with two y-axes
    ax1 = axes[1]
    ax1.set_ylabel('Temperature (K)', color='red')
    ax1.plot(energies['time_ps'], energies['T_K'], color='red', label='Temperature', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.axhline(300, color='black', linestyle='--', alpha=0.5, label="Target T (300K)")

    # Plot dPE on secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('$\Delta$PE per Update (kJ/mol)', color='blue')
    ax2.plot(updates['time_ps'], updates['dPE'], color='blue', marker='o', markersize=2, linestyle='None', label='$\Delta$PE (Energy Injection)', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Temperature Runaway correlated with Energy Injection')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Plot 3: Charge Updates Magnitude ---
    ax = axes[2]
    ax.plot(updates["time_ps"], updates["dq_max"], 'g-', label="Max $\Delta q$ (+)", alpha=0.7)
    ax.plot(updates["time_ps"], updates["dq_min"], 'b-', label="Min $\Delta q$ (-)", alpha=0.7)
    ax.fill_between(updates["time_ps"], updates["dq_min"], updates["dq_max"], color='gray', alpha=0.1)
    ax.set_ylabel("Charge Change ($\Delta e$)")
    ax.set_xlabel("Simulation Time (ps)")
    ax.set_title("Magnitude of Charge Updates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show() # This command will open the plot in a standard Python environment

# --- Command-line Argument Handling ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot molecular dynamics energetics data from two files.")

    # Add the --ene argument for the energies file path
    parser.add_argument(
        '--ene',
        type=str,
        required=True,
        help="Path to the energies data file (e.g., nvtmd_energies.dat)."
    )

    # Add the --upd argument for the updates file path
    parser.add_argument(
        '--upd',
        type=str,
        required=True,
        help="Path to the updates data file (e.g., nvtmd_updates.dat)."
    )

    args = parser.parse_args()

    # Call the plotting function with the paths provided via command-line
    plot_energetics(args.ene, args.upd)
