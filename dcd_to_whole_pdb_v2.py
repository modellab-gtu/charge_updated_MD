#!/usr/bin/env python3
"""
Convert OpenMM DCD to 'whole' trajectory for GROMACS and
optionally use the *last* box in the PDB CRYST1 line,
and save final-frame PDB/GRO.

Usage example:

    python dcd_to_whole_traj.py \
        --dcd traj.dcd \
        --top system.pdb \
        --out-pdb traj_whole.pdb \
        --pdb-box-from last \
        --out-xtc traj_whole.xtc \
        --final-pdb final_frame.pdb \
        --final-gro final_frame.gro
"""

import argparse
import mdtraj as md


def main():
    parser = argparse.ArgumentParser(
        description="Convert DCD to whole trajectory using MDTraj."
    )
    parser.add_argument("--dcd", required=True,
                        help="Input DCD trajectory from OpenMM.")
    parser.add_argument("--top", required=True,
                        help="Topology file (PDB, PRMTOP, GRO, PSF, etc.).")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride for reading frames (default: 1 = all).")

    parser.add_argument("--out-pdb", default=None,
                        help="Output PDB trajectory (multi-model).")
    parser.add_argument("--pdb-box-from", choices=["first", "last"],
                        default="first",
                        help=("Which frame's box to use in PDB CRYST1 "
                              "(since PDB supports only one box). "
                              "Default: first"))
    parser.add_argument("--out-xtc", default=None,
                        help="Output XTC trajectory (with per-frame box).")

    parser.add_argument("--final-pdb", default=None,
                        help="Save last frame as separate PDB.")
    parser.add_argument("--final-gro", default=None,
                        help="Save last frame as separate GRO.")

    args = parser.parse_args()

    print(f"Loading trajectory: {args.dcd}")
    traj = md.load_dcd(args.dcd, top=args.top, stride=args.stride)

    print("Re-imaging molecules to make them whole...")
    traj = traj.image_molecules(inplace=False)

    print("Centering coordinates...")
    traj.center_coordinates()

    has_box = traj.unitcell_lengths is not None

    # ---- Control which box appears in PDB CRYST1 ----
    if has_box and args.out_pdb is not None:
        if args.pdb_box_from == "last":
            print("Setting all PDB box values to the LAST frame box...")
            last_lengths = traj.unitcell_lengths[-1].copy()
            last_angles = traj.unitcell_angles[-1].copy()
            traj.unitcell_lengths[:] = last_lengths
            traj.unitcell_angles[:] = last_angles
        else:
            print("Using FIRST frame box for PDB CRYST1 (default behavior).")

    # ---- Save multi-model PDB (trajectory) ----
    if args.out_pdb is not None:
        print(f"Saving multi-model PDB trajectory to: {args.out_pdb}")
        traj.save_pdb(args.out_pdb)

    # ---- Save XTC with per-frame box (if present) ----
    if args.out_xtc is not None:
        if not has_box:
            print("Warning: trajectory has no unit cell info; XTC will not "
                  "have meaningful box vectors.")
        print(f"Saving XTC trajectory to: {args.out_xtc}")
        traj.save_xtc(args.out_xtc)

    # ---- Save last frame as separate PDB/GRO with final box ----
    last_frame = traj[-1]

    if args.final_pdb is not None:
        print(f"Saving last frame as PDB to: {args.final_pdb}")
        last_frame.save_pdb(args.final_pdb)

    if args.final_gro is not None:
        print(f"Saving last frame as GRO to: {args.final_gro}")
        last_frame.save_gro(args.final_gro)

    print("Done.")


if __name__ == "__main__":
    main()

