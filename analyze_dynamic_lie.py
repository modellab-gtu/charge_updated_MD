import argparse
import sys
import os
import numpy as np
import pandas as pd
import mdtraj as md
import openmm as mm
from openmm import app, unit
from copy import deepcopy

# -----------------------------------------------------------------------------
# CONFIGURATION DEFAULTS
# -----------------------------------------------------------------------------
DEFAULT_TOPOLOGY = "MOL_solv.prmtop"
DEFAULT_XYZ = "nvtmd_snapshots.xyz"
LIGAND_RESNAME = "MOL"

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Dynamic vs Static LIE from OpenMM-AIMNet2 outputs.")
    parser.add_argument("--top", default=DEFAULT_TOPOLOGY, help="Topology file (prmtop, gro, pdb)")
    parser.add_argument("--xyz", default=DEFAULT_XYZ, help="XYZ file with charges (nvtmd_snapshots.xyz)")
    parser.add_argument("--ligand", default=LIGAND_RESNAME, help="Residue name of the ligand (default: MOL)")
    parser.add_argument("--out", default="lie_analysis_results.csv", help="Output CSV filename")
    parser.add_argument("--cutoff", type=float, default=1.0, help="Nonbonded cutoff in nm (default: 1.0)")
    return parser.parse_args()

def get_force_by_type(system, force_type):
    """Helper to get a specific force object from a system."""
    for f in system.getForces():
        if isinstance(f, force_type):
            return f
    return None

def load_xyz_snapshots(filename):
    """
    Parses the custom XYZ format produced by the OpenMM runner.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    frames = []
    i = 0
    while i < len(lines):
        try:
            line = lines[i].strip()
            if not line: # skip empty lines
                i += 1
                continue
            natoms = int(line)
        except ValueError:
            break
            
        header = lines[i+1]
        
        # Parse Box info from header: box=[a,b,c]
        import re
        box_match = re.search(r'box=\[([\d\.]+),([\d\.]+),([\d\.]+)\]', header)
        if box_match:
            # XYZ header is in Angstroms based on runner code, OpenMM needs Nanometers
            box_vectors = [float(box_match.group(1))/10.0, 
                           float(box_match.group(2))/10.0, 
                           float(box_match.group(3))/10.0] 
        else:
            box_vectors = None

        atom_lines = lines[i+2 : i+2+natoms]
        
        coords = []
        charges = []
        
        for line in atom_lines:
            parts = line.split()
            # El x y z q
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            q = float(parts[4])
            coords.append([x, y, z])
            charges.append(q)
            
        coords = np.array(coords) / 10.0 # Convert Angstrom to NM for OpenMM
        
        frames.append({
            'coords': coords * unit.nanometer,
            'charges': np.array(charges),
            'box': box_vectors
        })
        
        i += (natoms + 2)
        
    print(f"Loaded {len(frames)} snapshots from {filename}")
    return frames

def measure_energy_components(sim, particle_indices, charges, sigmas, epsilons):
    """
    Applies the specific parameters (q, sigma, epsilon) to the particles in the simulation
    and returns (Coulombic, LJ) energies.
    
    To separate Coul/LJ, we:
    1. Calculate Total Nonbonded Energy (E_tot = Coul + LJ)
    2. Set all charges to 0, calculate Energy (E_lj)
    3. Coul = E_tot - E_lj
    """
    nb = get_force_by_type(sim.system, mm.NonbondedForce)
    n_particles = nb.getNumParticles()

    # 1. APPLY TARGET PARAMETERS & CALC TOTAL ENERGY
    for idx, q, sig, eps in zip(particle_indices, charges, sigmas, epsilons):
        nb.setParticleParameters(idx, q*unit.elementary_charge, sig, eps)
    nb.updateParametersInContext(sim.context)
    
    state_tot = sim.context.getState(getEnergy=True)
    e_tot = state_tot.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    # 2. ZERO CHARGES & CALC LJ ENERGY
    # We keep sigma/epsilon the same, but set q=0 for ALL particles to isolate LJ
    for idx, sig, eps in zip(particle_indices, sigmas, epsilons):
        nb.setParticleParameters(idx, 0.0*unit.elementary_charge, sig, eps)
    nb.updateParametersInContext(sim.context)

    state_lj = sim.context.getState(getEnergy=True)
    e_lj = state_lj.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    # 3. DERIVE COULOMBIC
    e_coul = e_tot - e_lj
    
    return e_coul, e_lj

def main():
    args = parse_args()
    
    # 1. Load Topology & Create System
    print(f"Loading topology: {args.top}")
    if args.top.endswith('.prmtop'):
        top_file = app.AmberPrmtopFile(args.top)
        topology = top_file.topology
        # Create system from prmtop
        system = top_file.createSystem(
            nonbondedMethod=app.PME, 
            nonbondedCutoff=args.cutoff*unit.nanometer,
            constraints=app.HBonds, 
            rigidWater=True
        )
    elif args.top.endswith('.gro'):
        # For .gro we usually need a top file, but if user provides .gro as --top, 
        # we can only read topology, not create system with forces unless .top is also known.
        # Assuming typical GROMACS usage: --top topol.top
        print("Error: For GROMACS, please provide a .top file for --top, not .gro")
        sys.exit(1)
    else:
        # PDB or other
        top_file = app.PDBFile(args.top)
        topology = top_file.topology
        # PDB doesn't have forcefield params, so we can't createSystem easily without xml
        print("Error: PDB file provided without ForceField source. Use .prmtop or .top")
        sys.exit(1)

    # 2. Identify Indices
    print("Identifying atom groups...")
    md_top = md.Topology.from_openmm(topology)
    ligand_indices = md_top.select(f"resname {args.ligand}")
    solvent_indices = md_top.select(f"not resname {args.ligand}")
    
    if len(ligand_indices) == 0:
        print(f"Error: No atoms found for ligand resname '{args.ligand}'")
        sys.exit(1)
        
    print(f"  Complex atoms: {md_top.n_atoms}")
    print(f"  Ligand atoms:  {len(ligand_indices)}")
    print(f"  Solvent atoms: {len(solvent_indices)}")
    
    all_indices = np.arange(md_top.n_atoms)

    # 3. Initialize Simulation
    # We use one simulation and update parameters/positions
    integrator = mm.VerletIntegrator(0.001)
    platform = mm.Platform.getPlatformByName('CPU')
    sim = app.Simulation(topology, system, integrator, platform)
    
    # 4. Get BASE Parameters (Sigma, Epsilon, Static Charges) from prmtop
    # We need these to restore/mask interactions
    nb = get_force_by_type(system, mm.NonbondedForce)
    base_sigmas = []
    base_epsilons = []
    base_charges_static = []
    
    for i in range(nb.getNumParticles()):
        chg, sig, eps = nb.getParticleParameters(i)
        base_charges_static.append(chg.value_in_unit(unit.elementary_charge))
        base_sigmas.append(sig)
        base_epsilons.append(eps)
    
    base_charges_static = np.array(base_charges_static)
    base_sigmas = np.array(base_sigmas)  # Quantity array? No, list of Quantities
    base_epsilons = np.array(base_epsilons)

    # 5. Load Trajectory
    snapshots = load_xyz_snapshots(args.xyz)
    results = []
    
    print("Starting Analysis Loop (Masking Method)...")

    for frame_idx, frame in enumerate(snapshots):
        # Update Box & Positions
        if frame['box']:
            a = mm.Vec3(frame['box'][0], 0, 0) * unit.nanometer
            b = mm.Vec3(0, frame['box'][1], 0) * unit.nanometer 
            c = mm.Vec3(0, 0, frame['box'][2]) * unit.nanometer
            sim.context.setPeriodicBoxVectors(a,b,c)
        sim.context.setPositions(frame['coords'])
        
        current_q_dyn = frame['charges'] # From AIMNet2
        
        # Helper to simplify calls
        def get_components(mask_indices, q_set):
            # Create masked arrays
            # We want: 
            #  - q = q_set[i] for i in mask_indices, else 0
            #  - eps = eps[i] for i in mask_indices, else 0 (Turn off LJ)
            #  - sig = sig[i] (doesn't matter if eps 0, but keep it)
            
            # 1. Start with zeros
            q_active = np.zeros_like(q_set)
            eps_active = [0.0*unit.kilojoule_per_mole] * len(base_epsilons)
            
            # 2. Fill in active indices
            # Numpy advanced indexing for q
            q_active[mask_indices] = q_set[mask_indices]
            
            # List comprehension/loop for units (safer than numpy object arrays with units)
            # Note: base_epsilons is a list of Quantity
            for i in mask_indices:
                eps_active[i] = base_epsilons[i]
                
            return measure_energy_components(sim, all_indices, q_active, base_sigmas, eps_active)

        # --- A. STATIC CALCS ---
        # 1. Complex (All atoms active)
        Ec_comp, Elj_comp = get_components(all_indices, base_charges_static)
        # 2. Ligand (Ligand active, Solvent zeroed)
        Ec_lig, Elj_lig   = get_components(ligand_indices, base_charges_static)
        # 3. Solvent (Solvent active, Ligand zeroed)
        Ec_sol, Elj_sol   = get_components(solvent_indices, base_charges_static)
        
        delta_coul_static = Ec_comp - (Ec_lig + Ec_sol)
        delta_lj_static   = Elj_comp - (Elj_lig + Elj_sol)

        # --- B. DYNAMIC CALCS ---
        # 1. Complex
        Ec_comp_d, Elj_comp_d = get_components(all_indices, current_q_dyn)
        # 2. Ligand
        Ec_lig_d, Elj_lig_d   = get_components(ligand_indices, current_q_dyn)
        # 3. Solvent
        Ec_sol_d, Elj_sol_d   = get_components(solvent_indices, current_q_dyn)

        delta_coul_dyn = Ec_comp_d - (Ec_lig_d + Ec_sol_d)
        delta_lj_dyn   = Elj_comp_d - (Elj_lig_d + Elj_sol_d)

        results.append({
            'Frame': frame_idx,
            'Delta_Coul_Static': delta_coul_static,
            'Delta_LJ_Static': delta_lj_static,
            'Delta_Coul_Dyn': delta_coul_dyn,
            'Delta_LJ_Dyn': delta_lj_dyn,
            'LIE_Static': delta_coul_static + delta_lj_static,
            'LIE_Dyn': delta_coul_dyn + delta_lj_dyn
        })

        if frame_idx % 10 == 0:
            sys.stdout.write(f"\rProcessed frame {frame_idx}/{len(snapshots)}")
            sys.stdout.flush()

    # Final Stats
    print("\nProcessing complete.")
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    
    avg = df.mean()
    std = df.std()
    
    # Construct summary table
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append(f"FINAL RESULTS (Average Interaction Energies in kJ/mol) over {len(df)} frames")
    summary_lines.append("="*80)
    summary_lines.append(f"{'Term':<20} | {'Static (Fixed Q)':<20} | {'Dynamic (AIMNet2)':<20}")
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'Coulombic':<20} | {avg['Delta_Coul_Static']:8.2f} +/- {std['Delta_Coul_Static']:5.2f} | {avg['Delta_Coul_Dyn']:8.2f} +/- {std['Delta_Coul_Dyn']:5.2f}")
    summary_lines.append(f"{'Lennard-Jones':<20} | {avg['Delta_LJ_Static']:8.2f} +/- {std['Delta_LJ_Static']:5.2f} | {avg['Delta_LJ_Dyn']:8.2f} +/- {std['Delta_LJ_Dyn']:5.2f}")
    summary_lines.append(f"{'Total (LIE)':<20} | {avg['LIE_Static']:8.2f} +/- {std['LIE_Static']:5.2f} | {avg['LIE_Dyn']:8.2f} +/- {std['LIE_Dyn']:5.2f}")
    summary_lines.append("="*80)
    
    summary_str = "\n".join(summary_lines)
    
    # Print to stdout
    print("\n" + summary_str)
    
    # Save to file
    summary_file = "lie_summary.dat"
    with open(summary_file, "w") as f:
        f.write(summary_str + "\n")
        
    print(f"Detailed frame-by-frame data saved to: {args.out}")
    print(f"Summary statistics saved to: {summary_file}")

if __name__ == "__main__":
    main()
