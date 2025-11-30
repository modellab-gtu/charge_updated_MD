Dynamic Charge Molecular Dynamics: A Comprehensive Tutorial

This tutorial provides a step-by-step guide to running Molecular Dynamics (MD) simulations where atomic partial charges are updated on-the-fly using the AIMNet2 neural network potential. This approach allows for capturing electronic polarization and induced-fit effects, which are typically absent in classical fixed-charge force fields.

The workflow consists of three main phases:

Preparation: Setting up the system topology and coordinates.

Simulation: Running the dynamic charge MD engine.

Analysis: Calculating interaction energies, structural properties, and visualizing charge dynamics.

Phase 1: System Preparation

Before starting the dynamic simulation, you must have a standard equilibrated system prepared using tools like AmberTools (tleap) or GROMACS.

1. Prepare Topology and Coordinates

Run the provided preparation script to generate the necessary input files.

./prep.sh


What this does:

Solvates your ligand in a water box.

Adds ions to neutralize the system.

Generates the topology file (MOL_solv.prmtop or topol.top) and coordinate file (MOL_solv.inpcrd or conf.gro).

Note: Ensure the ligand residue name in your topology (e.g., "MOL") matches the ligand-resname parameter in the configuration file.

Phase 2: Running the Simulation

The simulation is driven by the openmm_charge_update_runner_v20.py script, which is controlled by a YAML configuration file (v20.yaml).

1. Configuration Optimization (v20.yaml)

The v20.yaml file controls the physics and logic of the simulation. You must optimize these parameters for your specific scientific goals.

Critical Parameters

Parameter

Recommended Value

Description

constraints

HBonds

Constrains bonds involving hydrogen. Essential for stability with dt=0.002 or 0.0005.

rigid-water

False

False = Flexible, polarizable water (allows angle bending). True = Standard rigid water (TIP3P).

dt

0.0005

Time step in picoseconds (ps). 



• Use 0.0005 (0.5 fs) if rigid-water: False (Required for stability). 



• Use 0.002 (2.0 fs) if rigid-water: True.

dq_max

0.3

Maximum allowed charge change per atom per update. Prevents non-physical "shocks" to the system.

alpha

0.5

Smoothing factor (Exponential Moving Average). 



• 0.2: Slow, smooth evolution. 



• 0.8: Fast, reactive updates.

update-every

1

Update charges every N segments. 1 provides the most accurate polarization response.

solute-charge-scale

1.0

Scaling factor for ligand charges. Usually kept at 1.0.

solvent-charge-scale

1.5 - 2.0

Scales water charges. AIMNet2 predicts gas-phase charges; scaling mimics bulk liquid polarization (dipole moment).

q_abs_max

2.0

Hard cap on the absolute partial charge of any single atom to prevent runaway values.

Versatility & Modes

Classical Fixed-Charge Mode: Set update-every: 1000000 or dq_max: 0.0. The simulation will run using standard force field charges without updates.

Ensemble (NVT vs NPT):

NVT: Set barostat: False (Constant Volume).

NPT: Set barostat: True and provide a pressure value (Constant Pressure).

Warmup: During warmup_segments, charges are NOT updated. This allows the system to equilibrate geometrically before the dynamic potential is activated.

2. Execution

Run the simulation using your configured YAML file:

python openmm_charge_update_runner_v20.py --config v20.yaml


3. Monitoring the Simulation

While the simulation runs, watch the console output and logs:

Temperature: Ensure it fluctuates around your target (e.g., 300K). If it skyrockets (>500K), the system is exploding. Solution: Decrease dt or increase friction.

Box Volume: In NPT, ensure the volume stabilizes. If it expands indefinitely, your solvent-charge-scale might be too high (charges too repulsive).

Phase 3: Post-Processing & Analysis

Once the simulation completes, use the analysis suite to extract scientific insights.

1. Interaction Energy Analysis (LIE)

Calculate the Linear Interaction Energy (LIE) between the ligand and solvent. This decomposes the interaction into Coulombic and Lennard-Jones terms, comparing the Dynamic (AIMNet2) charges against the Static (Force Field) charges.

python analyze_dynamic_lie.py --top MOL_solv.prmtop --xyz nvtmd_snapshots.xyz --ligand MOL


Output: lie_summary.dat (Averages) and lie_analysis_results.csv (Time series).

Interpretation: A more negative Dynamic $\Delta E_{Coul}$ compared to Static suggests the ligand is effectively polarizing the environment to form stronger interactions.

2. Simulation Quality Check

Visualize the stability and performance of your simulation.

# Plot Temperature, Density, and Potential Energy
python plot_energies.py --input nvtmd_energies.dat

# Plot Charge Update Statistics (Drift, Magnitude, Energy Conservation)
python plot_updates.py --input nvtmd_updates.dat


Check: Verify charge_sum_e remains close to integer values. Large values in shift or dPE (Energy Change) indicate alpha or dq_max may need tuning.

3. Trajectory Processing

The raw .dcd trajectory often has molecules split across periodic boundaries. Convert it to a "whole" PDB for visualization.

python dcd_to_whole_pdb_v2.py --top MOL_solv.prmtop --dcd nvtmd_traj.dcd --out clean_traj.pdb --stride 10


4. Structural & Property Analysis

Run the advanced analysis script to calculate structural properties and generate visualization files.

python analyze_dynamic_properties_v2.py --top MOL_solv.prmtop --xyz nvtmd_snapshots.xyz --ligand MOL --out-prefix sim1


Key Outputs & Interpretation:

sim1_water_angle_hist.png:

Rigid Water: Shows a sharp spike (delta function) at ~104.5°.

Flexible Water: Shows a Gaussian (bell-shaped) distribution. This confirms that water molecules are flexible and polarizing geometrically.

sim1_rdf.png: Radial Distribution Function showing the structure of the hydration shells.

sim1_water_bond_hist.png: Should show a sharp spike if constraints: HBonds was used (recommended).

5. Visualization

A. Atomic Charge Movie

The file sim1_charge_vis.pdb (or nvtmd_snapshots.pdb) contains partial charges in the B-factor column.

Open in VMD.

Set Drawing Method to VDW or Licorice.

Set Coloring Method to Beta.

Set Trajectory tab color scale range to -1.0 to 1.0 (Red to Blue).

Play the movie to see charges fluctuate on the atoms!

B. Charge Density Surface (Volumetric)

To visualize the "average charge cloud" around your ligand:

Ensure you ran analyze_dynamic_properties_v2.py.

Run the generated VMD script:

vmd -e visualize.tcl

