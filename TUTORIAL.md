# Dynamic Charge Molecular Dynamics: A Comprehensive Tutorial

This tutorial provides a step-by-step guide to running Molecular Dynamics (MD) simulations where atomic partial charges are updated on-the-fly using the **AIMNet2** neural network potential. This approach allows you to capture electronic polarization and induced-fit effects that are typically absent in classical fixed-charge force fields.

The workflow consists of three main phases:

1. **Preparation** – Setting up the system topology and coordinates  
2. **Simulation** – Running the dynamic charge MD engine  
3. **Analysis** – Calculating interaction energies, structural properties, and visualizing charge dynamics  

---

## Phase 1: System Preparation

Before starting the dynamic simulation, you must have a standard equilibrated system prepared using tools like **AmberTools (tleap)** or **GROMACS**.

### 1. Prepare Topology and Coordinates

Run the provided preparation script to generate the necessary input files:

```bash
./prep.sh
```

> [!NOTE]
> This script typically solvates your ligand in a water box, adds ions to neutralize the system, and generates the topology (`.prmtop` or `.top`) and coordinate (`.inpcrd` or `.gro`) files.

> [!IMPORTANT]
> Ensure the ligand residue name in your topology (for example, `MOL`) matches the `ligand-resname` parameter in your configuration file.

---

## Phase 2: Running the Simulation

The simulation is driven by the `openmm_charge_update_runner_v20.py` script, controlled by a YAML configuration file (for example, `v20.yaml`).

### 1. Configuring `v20.yaml`

The YAML file controls the physics and logic of the simulation. You must optimize these parameters for your specific scientific goals.

#### Critical Parameters

| Parameter               | Recommended value  | Description |
| ----------------------- | ----------------- | ----------- |
| `constraints`           | `HBonds`          | Constrains bonds involving hydrogen. Essential for stability with `dt = 0.002` or `0.0005`. |
| `rigid-water`           | `false`           | `false` = flexible, polarizable water (allows angle bending). `true` = standard rigid water (TIP3P). |
| `dt`                    | `0.0005`          | Time step in picoseconds. Use `0.0005` (0.5 fs) if `rigid-water: false`. Use `0.002` (2.0 fs) if `rigid-water: true`. |
| `dq_max`                | `0.3`             | Maximum allowed charge change per atom per update. Prevents non-physical “shocks” to the system. |
| `alpha`                 | `0.5`             | Smoothing factor (exponential moving average). Smaller values (e.g., `0.2`) give slow, smooth evolution; larger values (e.g., `0.8`) give fast, reactive updates. |
| `update-every`          | `1`               | Update charges every *N* segments. `1` provides the most accurate polarization response. |
| `solute-charge-scale`   | `1.0`             | Scaling factor for ligand charges. Usually kept at `1.0`. |
| `solvent-charge-scale`  | `1.5–2.0`         | Scales water charges. AIMNet2 predicts gas-phase charges; scaling mimics bulk liquid polarization. |
| `q_abs_max`             | `2.0`             | Hard cap on the absolute partial charge of any single atom to prevent runaway values. |

#### Modes and Options

- **Classical fixed-charge mode**  
  Set `update-every: 1000000` **or** `dq_max: 0.0`. The simulation will run using standard force-field charges without updates.

- **Ensemble choice (NVT vs NPT)**  
  - **NVT**: set `barostat: false` (constant volume).  
  - **NPT**: set `barostat: true` and provide a `pressure` value (constant pressure).

- **Warmup**  
  During `warmup_segments`, charges are **not** updated. This allows the system to equilibrate geometrically before the dynamic potential is activated.

### 2. Running the Simulation

Run the simulation using your configured YAML file:

```bash
python openmm_charge_update_runner_v20.py --config v20.yaml
```

### 3. Monitoring the Simulation

While the simulation runs, watch the console output and log files.

> [!WARNING]
> **Check temperature:** Ensure it fluctuates around your target (e.g., 300 K).  
> If it skyrockets (e.g., > 500 K), the system is exploding.  
> **Fix:** Decrease `dt` or increase `friction`.

> [!CAUTION]
> **Check box volume:** In NPT simulations, the volume should stabilize.  
> If it expands indefinitely, your `solvent-charge-scale` may be too high (charges too repulsive).

---

## Phase 3: Post-Processing & Analysis

Once the simulation completes, use the analysis suite to extract scientific insights.

### 1. Interaction Energy Analysis (LIE)

Calculate the Linear Interaction Energy (LIE) between the ligand and solvent. This decomposes the interaction into **Coulombic** and **Lennard-Jones** terms, comparing **Dynamic (AIMNet2)** charges against **Static (force-field)** charges.

```bash
python analyze_dynamic_lie.py   --top MOL_solv.prmtop   --xyz nvtmd_snapshots.xyz   --ligand MOL
```

**Outputs**

- `lie_summary.dat` – Average interaction energies  
- `lie_analysis_results.csv` – Time-series data  

**Interpretation**

- A more negative **Dynamic** LIE compared to **Static** suggests that the ligand is effectively polarizing its environment to form stronger interactions.

### 2. Simulation Quality Check

Visualize the stability and performance of your simulation:

```bash
# Plot temperature, density, and potential energy
python plot_energies.py --input nvtmd_energies.dat

# Plot charge update statistics (drift, magnitude, energy conservation)
python plot_updates.py --input nvtmd_updates.dat
```

> [!TIP]
> Verify `charge_sum_e` remains close to integer values. Large values in `shift` or `dPE` (energy change) indicate that `alpha` or `dq_max` may need tuning.

### 3. Trajectory Processing

The raw `.dcd` trajectory often has molecules split across periodic boundaries. Convert it to a “whole” PDB for visualization:

```bash
python dcd_to_whole_pdb_v2.py   --top MOL_solv.prmtop   --dcd nvtmd_traj.dcd   --out clean_traj.pdb
```

### 4. Structural & Property Analysis

Run the advanced analysis script to calculate structural properties and generate visualization files:

```bash
python analyze_dynamic_properties_v2.py   --top MOL_solv.prmtop   --xyz nvtmd_snapshots.xyz   --ligand MOL
```

#### Key Outputs & Interpretation

1. **`sim1_water_angle_hist.png`**  
   - *Rigid water*: a sharp spike (delta function) at ≈104.5°.  
   - *Flexible water*: a Gaussian (bell-shaped) distribution.  
   This confirms whether water molecules are flexible and polarizing geometrically.

2. **`sim1_rdf.png`**  
   - Radial distribution functions showing the structure of hydration shells around the solute.

3. **`sim1_water_bond_hist.png`**  
   - Should show a sharp spike if `constraints: HBonds` was used (recommended), confirming constrained bond lengths.

---

## Visualization

### A. Atomic Charge Movie

The file `sim1_charge_vis.pdb` (or `nvtmd_snapshots.pdb`) contains partial charges in the **B-factor** column.

1. Open the file in **VMD**.  
2. Set **Drawing Method** to *VDW* or *Licorice*.  
3. Set **Coloring Method** to *Beta*.  
4. In the *Trajectory* tab, set the color scale range to **−1.0 to 1.0** (e.g., red to blue).  
5. Play the movie to see charges fluctuate on the atoms over time.

### B. Charge Density Surface (Volumetric)

To visualize the “average charge cloud” around your ligand:

1. Ensure you have already run `analyze_dynamic_properties_v2.py`.  
2. Run the generated VMD script:

```bash
vmd -e visualize.tcl
```
