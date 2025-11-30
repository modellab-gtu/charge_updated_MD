import argparse
import sys
import os
import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib.pyplot as plt
from openmm import unit

# Try to import scipy for Gaussian smoothing (Charge Density)
try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DEFAULT_TOPOLOGY = "MOL_solv.prmtop"
DEFAULT_XYZ = "nvtmd_snapshots.xyz"
LIGAND_RESNAME = "MOL"

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze structural properties of Dynamic Charge MD.")
    parser.add_argument("--top", default=DEFAULT_TOPOLOGY, help="Topology file")
    parser.add_argument("--xyz", default=DEFAULT_XYZ, help="XYZ file with charges")
    parser.add_argument("--ligand", default=LIGAND_RESNAME, help="Ligand Residue Name")
    parser.add_argument("--out-prefix", default="analysis", help="Prefix for output images/files")
    parser.add_argument("--grid-spacing", type=float, default=1.0, help="Grid spacing for DX file (Angstroms)")
    return parser.parse_args()

def load_custom_xyz_to_mdtraj(xyz_filename, top_filename):
    print(f"Loading {xyz_filename}...")
    with open(xyz_filename, 'r') as f:
        lines = f.readlines()
    
    coords_list = []
    charges_list = []
    
    i = 0
    while i < len(lines):
        try:
            line = lines[i].strip()
            if not line: i+=1; continue
            natoms = int(line)
        except ValueError:
            break
            
        header = lines[i+1]
        
        # Parse Box info
        import re
        box_match = re.search(r'box=\[([\d\.]+),([\d\.]+),([\d\.]+)\]', header)
        box_vec = None
        if box_match:
            a = float(box_match.group(1))/10.0
            b = float(box_match.group(2))/10.0
            c = float(box_match.group(3))/10.0
            box_vec = np.array([a, b, c])
        
        atom_lines = lines[i+2 : i+2+natoms]
        frame_coords = []
        frame_charges = []
        
        for line in atom_lines:
            parts = line.split()
            frame_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            frame_charges.append(float(parts[4]))
            
        coords_list.append(np.array(frame_coords) / 10.0) # A -> nm
        charges_list.append(np.array(frame_charges))
        i += (natoms + 2)
        
    print(f"Loading topology {top_filename}...")
    topology_obj = None
    try:
        topology_obj = md.load_prmtop(top_filename)
    except Exception:
        pass
        
    if topology_obj is None:
        try:
            temp_traj = md.load(top_filename)
            topology_obj = temp_traj.topology
        except Exception as e:
            print(f"Error loading topology: {e}")
            sys.exit(1)
        
    xyz = np.array(coords_list)
    
    if box_vec is not None:
        unitcell_lengths = np.tile(box_vec, (len(xyz), 1))
        unitcell_angles = np.tile(np.array([90.0, 90.0, 90.0]), (len(xyz), 1))
        traj = md.Trajectory(xyz, topology_obj, unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)
        print("Unwrapping molecules (image_molecules)...")
        traj.image_molecules(inplace=True)
    else:
        print("Warning: No box found. Using topology box.")
        traj = md.Trajectory(xyz, topology_obj)
        
    return traj, charges_list

def analyze_hbonds(traj, ligand_resname):
    print("\n--- 1. Hydrogen Bond Analysis ---")
    ligand_atom_indices = traj.topology.select(f"resname {ligand_resname}")
    solvent_atom_indices = traj.topology.select(f"not resname {ligand_resname} and (water or resname HOH or resname SOL or resname WAT)")
    
    if len(ligand_atom_indices) == 0:
        print("No ligand atoms found.")
        return

    hbonds = md.baker_hubbard(traj, periodic=True)
    n_frames = traj.n_frames
    hb_solute_solvent = np.zeros(n_frames)
    hb_solvent_solvent = np.zeros(n_frames)
    lig_set = set(ligand_atom_indices)
    sol_set = set(solvent_atom_indices)
    
    for hbond in hbonds:
        frame_id, a1, a2 = hbond
        is_a1_lig = a1 in lig_set; is_a2_lig = a2 in lig_set
        is_a1_sol = a1 in sol_set; is_a2_sol = a2 in sol_set
        
        if (is_a1_lig and is_a2_sol) or (is_a1_sol and is_a2_lig):
            hb_solute_solvent[frame_id] += 1
        if (is_a1_sol and is_a2_sol):
            hb_solvent_solvent[frame_id] += 1

    print(f"Avg Solute-Solvent H-bonds: {np.mean(hb_solute_solvent):.2f}")
    print(f"Avg Solvent-Solvent H-bonds: {np.mean(hb_solvent_solvent):.2f}")

def analyze_rdf(traj, ligand_resname, prefix):
    print("\n--- 2. RDF & Hydration Shells ---")
    ligand_heavy = traj.topology.select(f"resname {ligand_resname} and mass > 2")
    water_oxygen = traj.topology.select(f"water and (name O or element O)")
    
    if len(ligand_heavy) == 0 or len(water_oxygen) == 0: return

    pairs_lo = traj.topology.select_pairs(selection1=ligand_heavy, selection2=water_oxygen)
    r_lo, g_r_lo = md.compute_rdf(traj, pairs_lo, r_range=(0, 1.2))
    
    neighbors = md.compute_neighbors(traj, 0.35, query_indices=ligand_heavy, haystack_indices=water_oxygen)
    avg_coordination = np.mean([len(f) for f in neighbors])
    print(f"Avg Ligand Hydration Shell (Waters within 3.5A): {avg_coordination:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(r_lo, g_r_lo, label=f'{ligand_resname}(Heavy) - Water(O)', linewidth=2)
    plt.xlabel('r (nm)'); plt.ylabel('g(r)')
    plt.title('Solute-Solvent RDF')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{prefix}_rdf.png", dpi=300); plt.close()

def analyze_rmsf(traj, ligand_resname, prefix):
    print("\n--- 3. RMSF Analysis ---")
    ligand_indices = traj.topology.select(f"resname {ligand_resname} and mass > 2")
    if len(ligand_indices) == 0: return

    traj_align = traj[:] 
    traj_align.superpose(traj_align, 0, atom_indices=ligand_indices)
    rmsf = md.rmsf(traj_align, traj_align, frame=0, atom_indices=ligand_indices, precentered=True)
    
    plt.figure(figsize=(8, 5))
    atom_names = [traj.topology.atom(i).name for i in ligand_indices]
    plt.bar(range(len(rmsf)), rmsf * 10.0)
    plt.xticks(range(len(rmsf)), atom_names, rotation=90)
    plt.ylabel('RMSF ($\AA$)'); plt.title(f'Ligand RMSF')
    plt.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(f"{prefix}_rmsf.png", dpi=300); plt.close()

def analyze_water_geometry(traj, prefix):
    print("\n--- 4. Water Geometry Analysis ---")
    oh_pairs = []; hoh_triplets = []
    water_resnames = ['HOH', 'SOL', 'WAT', 'TIP3', 'TIP4P']
    
    for res in traj.topology.residues:
        if res.name not in water_resnames: continue
        oxygens = [a.index for a in res.atoms if a.element.symbol == 'O']
        hydrogens = [a.index for a in res.atoms if a.element.symbol == 'H']
        if len(oxygens) == 1 and len(hydrogens) >= 2:
            oh_pairs.append([oxygens[0], hydrogens[0]])
            oh_pairs.append([oxygens[0], hydrogens[1]])
            hoh_triplets.append([hydrogens[0], oxygens[0], hydrogens[1]])
            
    if not oh_pairs: return

    dists_ang = md.compute_distances(traj, np.array(oh_pairs)).flatten() * 10.0
    angles_deg = np.degrees(md.compute_angles(traj, np.array(hoh_triplets)).flatten())
    
    print(f"O-H Bond: Mean={np.mean(dists_ang):.6f} A, Std={np.std(dists_ang):.6f}")
    print(f"H-O-H Angle: Mean={np.mean(angles_deg):.6f} deg, Std={np.std(angles_deg):.6f}")
    
    def get_bins(data):
        if np.max(data) - np.min(data) < 1e-5: return np.linspace(np.min(data)-0.001, np.max(data)+0.001, 50)
        return 100

    plt.figure(figsize=(6, 5))
    plt.hist(dists_ang, bins=get_bins(dists_ang), color='teal', alpha=0.7, density=True)
    plt.xlabel('O-H Bond Length ($\AA$)'); plt.title('Water O-H Bond Distribution')
    plt.savefig(f"{prefix}_water_bond_hist.png", dpi=300); plt.close()
    
    plt.figure(figsize=(6, 5))
    plt.hist(angles_deg, bins=get_bins(angles_deg), color='orange', alpha=0.7, density=True)
    plt.xlabel('H-O-H Angle (Degrees)'); plt.title('Water H-O-H Angle Distribution')
    plt.savefig(f"{prefix}_water_angle_hist.png", dpi=300); plt.close()

def export_charge_colored_pdb(traj, charges, filename, frame_idx=-1):
    print(f"\n--- 5. Exporting Charge-Colored PDB (Single Frame) ---")
    top = traj.topology
    xyz = traj.xyz[frame_idx] * 10.0
    qs = charges[frame_idx]
    
    with open(filename, 'w') as f:
        f.write("REMARK   GENERATED BY ANALYZE_DYNAMIC_PROPERTIES.PY\n")
        f.write("REMARK   B-FACTOR COLUMN = PARTIAL CHARGE\n")
        atom_idx = 1
        for atom in top.atoms:
            res = atom.residue
            x, y, z = xyz[atom.index]
            q = qs[atom.index]
            record = "ATOM  " if res.name not in ['HOH', 'SOL', 'WAT', 'MOL', 'LIG'] else "HETATM"
            name = f"{atom.name:>4s}"
            resname = f"{res.name:>3s}"
            chain = "A"
            resseq = (res.index + 1) % 9999
            f.write(f"{record:<6s}{atom_idx:>5d} {name} {resname} {chain}{resseq:>4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{q:6.2f}           {atom.element.symbol:>2s}\n")
            atom_idx += 1
        f.write("END\n")
    print(f"Saved {filename}")

def export_trajectory_pdb(traj, charges_list, filename, stride=1):
    """
    Exports the entire trajectory as a multi-model PDB where B-factors = Charges.
    Allows for dynamic visualization in VMD.
    """
    print(f"\n--- 6. Exporting Trajectory PDB (stride={stride}) ---")
    print("  This allows animating charges in VMD.")
    
    top = traj.topology
    n_frames = traj.n_frames
    
    with open(filename, 'w') as f:
        f.write("REMARK   GENERATED BY ANALYZE_DYNAMIC_PROPERTIES.PY\n")
        
        for i in range(0, n_frames, stride):
            f.write(f"MODEL     {i+1}\n")
            xyz = traj.xyz[i] * 10.0
            qs = charges_list[i]
            
            atom_idx = 1
            for atom in top.atoms:
                res = atom.residue
                x, y, z = xyz[atom.index]
                q = qs[atom.index]
                record = "ATOM  " if res.name not in ['HOH', 'SOL', 'WAT', 'MOL', 'LIG'] else "HETATM"
                name = f"{atom.name:>4s}"
                resname = f"{res.name:>3s}"
                chain = "A"
                resseq = (res.index + 1) % 9999
                f.write(f"{record:<6s}{atom_idx:>5d} {name} {resname} {chain}{resseq:>4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{q:6.2f}           {atom.element.symbol:>2s}\n")
                atom_idx += 1
            f.write("ENDMDL\n")
            
            if i % 100 == 0:
                sys.stdout.write(f"\r  Writing frame {i}/{n_frames}...")
                sys.stdout.flush()
                
    print(f"\nSaved {filename}")

def export_averaged_charge_density_dx(traj, charges_list, filename, ligand_resname, spacing=1.0, padding=6.0):
    print(f"\n--- 7. Exporting Averaged Charge Density Surface (DX File) ---")
    if not HAS_SCIPY:
        print("[WARN] Scipy not found. Skipping DX generation.")
        return

    ligand_indices = traj.topology.select(f"resname {ligand_resname}")
    if len(ligand_indices) == 0:
        print(f"[ERROR] Ligand {ligand_resname} not found for alignment.")
        return

    print("  Aligning trajectory to Ligand for spatial averaging...")
    traj_aligned = traj[:]
    traj_aligned.superpose(traj_aligned, 0, atom_indices=ligand_indices)

    ref_coords = traj_aligned.xyz[0] * 10.0 # Angstroms
    lig_coords = ref_coords[ligand_indices]
    min_c = np.min(lig_coords, axis=0) - padding
    max_c = np.max(lig_coords, axis=0) + padding
    
    dims = np.ceil((max_c - min_c) / spacing).astype(int)
    grid_shape = tuple(dims)
    print(f"  Grid Bounds: {min_c} to {max_c}")
    print(f"  Grid Dim:    {grid_shape} (Spacing: {spacing} A)")
    
    print(f"  Accumulating charges over {traj.n_frames} frames...")
    accumulated_grid = np.zeros(dims)
    
    for i in range(traj.n_frames):
        coords = traj_aligned.xyz[i] * 10.0
        qs = charges_list[i]
        grid_frame, _ = np.histogramdd(coords, bins=dims, range=[(min_c[0], max_c[0]), (min_c[1], max_c[1]), (min_c[2], max_c[2])], weights=qs)
        accumulated_grid += grid_frame

    avg_grid = accumulated_grid / traj.n_frames
    
    print("  Applying Gaussian smoothing...")
    sigma_pixels = 1.0 / spacing 
    density_grid = gaussian_filter(avg_grid, sigma=sigma_pixels)

    print(f"  Writing {filename}...")
    with open(filename, 'w') as f:
        f.write("# Time-Averaged Charge Density Map\n")
        f.write("object 1 class gridpositions counts {0} {1} {2}\n".format(*grid_shape))
        f.write("origin {0} {1} {2}\n".format(*min_c))
        f.write("delta {0} 0 0\n".format(spacing))
        f.write("delta 0 {0} 0\n".format(spacing))
        f.write("delta 0 0 {0}\n".format(spacing))
        f.write("object 2 class gridconnections counts {0} {1} {2}\n".format(*grid_shape))
        f.write("object 3 class array type double rank 0 items {0} data follows\n".format(np.prod(grid_shape)))
        
        flat_data = density_grid.flatten()
        for i in range(0, len(flat_data), 3):
            chunk = flat_data[i:i+3]
            line = " ".join([f"{val:.6e}" for val in chunk])
            f.write(line + "\n")
            
        f.write('attribute "dep" string "positions"\n')
        f.write('object "regular positions regular connections" class field\n')
        f.write('component "positions" value 1\n')
        f.write('component "connections" value 2\n')
        f.write('component "data" value 3\n')
    print("  Done.")

def generate_vmd_script(script_name, avg_pdb_name, avg_dx_name, traj_pdb_name):
    print(f"\n--- 8. Generating VMD Visualization Script ---")
    tcl_content = f"""
# VMD Visualization Script
# Generated automatically

# --- VISUALIZATION 1: AVERAGE CHARGE MAP (Static) ---
# Use this for publication figures
mol new {avg_pdb_name} type pdb
mol modstyle 0 0 VDW 0.6 12.0
mol modcolor 0 0 Beta
mol scaleminmax 0 0 -1.0 1.0
color scale method RWB
mol rename 0 "Average Structure"

# Load Average DX Map
if {{ [file exists {avg_dx_name}] }} {{
    mol addfile {avg_dx_name} type dx
    
    # Red Cloud (Negative)
    mol addrep 0
    mol modstyle 1 0 Isosurface -0.05 0 0 0 1 1
    mol modcolor 1 0 ColorID 1 
    mol modmaterial 1 0 Transparent
    
    # Blue Cloud (Positive)
    mol addrep 0
    mol modstyle 2 0 Isosurface 0.05 0 0 0 1 1
    mol modcolor 2 0 ColorID 0 
    mol modmaterial 2 0 Transparent
}}

# --- VISUALIZATION 2: DYNAMIC CHARGES (Movie) ---
# Use this to see fluctuations (Turn off molecule 0 to see this)
if {{ [file exists {traj_pdb_name}] }} {{
    mol new {traj_pdb_name} type pdb
    mol rename 1 "Dynamic Trajectory"
    mol off 1
    
    # Atoms colored by dynamic charge
    mol modstyle 0 1 VDW 0.6 12.0
    mol modcolor 0 1 Beta
    mol scaleminmax 0 1 -1.0 1.0
    
    # Dynamic Volmap (Calculated inside VMD)
    # Uncomment next lines to generate dynamic cloud on-the-fly (Slow!)
    # volmap density [atomselect 1 "all"] -res 1.0 -weight beta -allframes -combine single -mol 1
}}

axes location Off
display projection Orthographic
display rendermode GLSL
    """
    with open(script_name, 'w') as f:
        f.write(tcl_content)
    print(f"Saved {script_name}")

def main():
    args = parse_args()
    traj, charges_list = load_custom_xyz_to_mdtraj(args.xyz, args.top)
    print(f"Loaded {traj.n_frames} frames.")
    
    analyze_hbonds(traj, args.ligand)
    analyze_rdf(traj, args.ligand, args.out_prefix)
    analyze_rmsf(traj, args.ligand, args.out_prefix)
    analyze_water_geometry(traj, args.out_prefix)
    
    # Files
    pdb_ref_name = f"{args.out_prefix}_charge_vis_ref.pdb"
    pdb_traj_name = f"{args.out_prefix}_charge_traj.pdb"
    dx_name = f"{args.out_prefix}_avg_charge_density.dx"
    
    # 1. Export Reference (Single Frame)
    export_charge_colored_pdb(traj, charges_list, filename=pdb_ref_name, frame_idx=0)
    
    # 2. Export Trajectory (For Movie)
    # Stride 10 to keep file size manageable for visualization
    export_trajectory_pdb(traj, charges_list, filename=pdb_traj_name, stride=10)
    
    # 3. Export Average DX (For Science/Fairness)
    export_averaged_charge_density_dx(traj, charges_list, filename=dx_name, ligand_resname=args.ligand, spacing=args.grid_spacing)
    
    # 4. VMD Script
    generate_vmd_script("visualize.tcl", pdb_ref_name, dx_name, pdb_traj_name)
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
