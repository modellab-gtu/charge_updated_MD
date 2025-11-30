# Dynamic-Charge Molecular Dynamics with AIMNet2

On-the-fly **partial charge updating** for Molecular Dynamics (MD) simulations using **OpenMM** and **AIMNet2**, with a full post-processing and visualization toolkit.

This repository provides:

- A segmented MD engine that updates atomic charges from AIMNet2 during the simulation  
- An Amber-based preparation pipeline for solvated ligand systems  
- Analysis tools for **LIE (Linear Interaction Energy)**, structural descriptors, and charge visualization  
- Plotting utilities for diagnosing stability, energy injection, and charge conservation  

For detailed usage and background, see:

- 📘 **[TUTORIAL.md](./TUTORIAL.md)** – step-by-step dynamic-charge MD workflow  
- 🛠 **[INSTALLATION.md](./INSTALLATION.md)** – full environment setup instructions  

---

## Repository Structure

- `openmm_charge_update_runner_v20.py` – main OpenMM + AIMNet2 dynamic-charge MD driver  
- `prep.sh` – AmberTools-based system builder (GAFF2 ligand + TIP3P water box)  
- `analyze_dynamic_lie.py` – compute Dynamic vs Static LIE (Coulomb + LJ) from snapshots  
- `analyze_dynamic_properties_v2.py` – structural and charge-based analysis (RDF, H-bonds, water geometry, charge-density maps, charge-colored PDBs)  
- `dcd_to_whole_pdb_v2.py` – convert OpenMM DCD to “whole” trajectory (PDB/XTC) and save final-frame structures  
- `plot_energies.py` – combined energy / temperature / ΔPE visualization from `energies.dat` and `updates.dat`  
- `plot_updates.py` – diagnostics of charge updates and net charge drift from `updates.dat`  
- `environment.yml` – Conda environment specification (OpenMM, PyTorch, AIMNet2, MDTraj, etc.)  
- `v20_reference.yaml` – example configuration file for `openmm_charge_update_runner_v20.py`  
- `TUTORIAL.md` – dynamic charge MD tutorial (usage, workflow, interpretation)  
- `INSTALLATION.md` – environment setup (conda, OpenMM, PyTorch, AIMNet2)  

---

## Installation

For full installation details (Conda, PyTorch, AIMNet2), see **[INSTALLATION.md](./INSTALLATION.md)**.

### Quick Start (recommended)

```bash
# Create environment from provided environment.yml
conda env create -f environment.yml

# Activate environment
conda activate openmm-cpu    # or the name given in environment.yml
```

---

## Typical Workflow

1. **System build (AmberTools)**
2. **Dynamic-charge MD (OpenMM + AIMNet2)**
3. **Post-processing and analysis**
4. **Visualization (VMD / plots)**

### 1. System Preparation

Use `prep.sh` to generate an Amber GAFF2 ligand in TIP3P water and create `MOL_solv.prmtop` / `MOL_solv.inpcrd`:

```bash
bash prep.sh
```

This script:

- Runs **antechamber** + **parmchk2** for the ligand  
- Builds a solvated box with **tleap**  
- Outputs `MOL_solv.prmtop` and `MOL_solv.inpcrd` for use in OpenMM  

### 2. Dynamic-Charge MD

The main engine is `openmm_charge_update_runner_v20.py`. It:

- Loads Amber (`.prmtop` + `.inpcrd`) or GROMACS (`.top` + `.gro`) systems  
- Builds an OpenMM `System` with configurable **constraints**, **rigid water**, and **nonbonded method**  
- Periodically calls AIMNet2 to update charges for all atoms  
- Smooths and caps charge changes (`alpha`, `dq_max`, `q_abs_max`)  
- Supports NVT warmup and optional NPT production with Monte Carlo barostat  
- Writes:
  - `traj.dcd` – DCD trajectory  
  - `energies.dat` – energies / box / density / total charge (TSV)  
  - `updates.dat` – per-update diagnostics (ΔPE, Δq stats, charge sum)  
  - Optional `XYZ`/`PDB` snapshots with charges in coordinates/B-factor  

Run with a YAML config (e.g. `v20_reference.yaml`):

```bash
python openmm_charge_update_runner_v20.py --config v20_reference.yaml
```

Key YAML options (see `TUTORIAL.md` for explanations):

- `prmtop`, `inpcrd` OR `top`, `gro`  
- `constraints`, `rigid-water`, `nonbonded-cutoff`  
- `segments`, `steps-per-seg`, `warmup-segments`, `dt`, `temperature`, `friction`  
- `barostat`, `pressure`, `barostat-interval`  
- `update-every`, `alpha`, `dq-max`, `q-abs-max`, `solute-charge-scale`, `solvent-charge-scale`  
- `xyz-with-charges`, `pdb-with-charges`, `energies-dat`, `updates-dat`, `dcd`  

Detailed parameter recommendations and troubleshooting are in **TUTORIAL.md**.

---

## Analysis & Visualization

After the run completes, use the analysis tools to extract interaction energies, structural metrics, and charge behavior.

### 3.1 Dynamic vs Static LIE

`analyze_dynamic_lie.py` compares **dynamic** (AIMNet2-based) and **static** (original force-field) interaction energies via LIE:

```bash
python analyze_dynamic_lie.py   --top MOL_solv.prmtop   --xyz nvtmd_snapshots.xyz   --ligand MOL   --out lie_analysis_results.csv   --cutoff 1.0
```

Outputs:

- `lie_analysis_results.csv` – per-frame Coulomb and LJ interaction energies (dynamic vs static)  
- `lie_summary.dat` (if implemented) – averages suitable for LIE-based ΔG estimates  

### 3.2 Structural & Charge-Related Properties

`analyze_dynamic_properties_v2.py` reads the custom XYZ with charges and computes:

- H-bond statistics (solute–solvent & solvent–solvent)  
- RDFs and hydration shell metrics  
- Ligand RMSF  
- Water geometry (O–H distances, H–O–H angle distributions)  
- Charge-colored single- and multi-model PDBs (charges in B-factor)  
- Optional time-averaged charge-density map in DX format (if SciPy available)  

Example:

```bash
python analyze_dynamic_properties_v2.py   --top MOL_solv.prmtop   --xyz nvtmd_snapshots.xyz   --ligand MOL   --out-prefix sim1
```

Key outputs include:

- `sim1_rdf.png`  
- `sim1_water_bond_hist.png`, `sim1_water_angle_hist.png`  
- `sim1_charge_vis.pdb` (single-frame PDB with charges as B-factors)  
- `sim1_charge_traj.pdb` (multi-model PDB for charge movies in VMD)  
- `sim1_charge_density.dx` (if DX export enabled)  

### 3.3 Trajectory Repair & Final Box

`dcd_to_whole_pdb_v2.py` converts OpenMM DCD files into whole, centered trajectories and optionally saves final-frame structures:

```bash
python dcd_to_whole_pdb_v2.py   --dcd traj.dcd   --top MOL_solv.prmtop   --stride 1   --out-pdb traj_whole.pdb   --pdb-box-from last   --out-xtc traj_whole.xtc   --final-pdb final_frame.pdb   --final-gro final_frame.gro
```

- `traj_whole.pdb` – multi-model PDB with a single CRYST1 box (first or last frame)  
- `traj_whole.xtc` – GROMACS-compatible trajectory with per-frame box  
- `final_frame.pdb` / `final_frame.gro` – last snapshot for inspection  

### 3.4 Energy & Charge Diagnostics

Use `plot_energies.py` to visualize global energy and temperature trends together with per-update ΔPE:

```bash
python plot_energies.py   --ene nvtmd_energies.dat   --upd nvtmd_updates.dat
```

Produces:

- A multi-panel plot:
  - Total & potential energy vs time  
  - Temperature vs time with overlaid ΔPE at each update  
  - Δq envelope (min/max) vs time  

Use `plot_updates.py` for a more detailed focus on charge updates and charge conservation:

```bash
python plot_updates.py   --upd nvtmd_updates.dat
```

Outputs:

- `charge_diagnostics.png` with:
  - ΔPE per update  
  - Δq min/max envelope and stability threshold (e.g., |Δq| ≈ 0.05 e)  
  - Total system charge vs time (zoomed to detect small drifts)  

---

## Documentation

- **[TUTORIAL.md](./TUTORIAL.md)** – conceptual overview, recommended settings, interpretive guides  
- **[INSTALLATION.md](./INSTALLATION.md)** – installation commands and verification snippet  

If you extend or modify the workflow (e.g., alternative charge models, different ML potentials), consider updating the tutorial and README accordingly.

---

## Citation

If you use this code in a publication, please cite:

- The underlying ML charge model (**AIMNet2**)  
- OpenMM & MDTraj  
- Any relevant methodology papers for your specific use case  

(You can add your own preprint/paper reference here once available.)

---

## License

Add your chosen license here, for example:

```text
MIT License
Copyright (c) 2025 ...
```
