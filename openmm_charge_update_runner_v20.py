#!/usr/bin/env python3
# openmm_charge_update_runner_v20.py
#
# Modifications from v19:
# - ADDED: Support for configurable constraints and rigid water via YAML/CLI.
#   --constraints (HBonds, AllBonds, HAngles, None)
#   --rigid-water (True/False)
# - UPDATED: Load functions and system rebuilder now use these arguments instead of hardcoded defaults.

import argparse, sys, os, json, re, ast
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

import openmm as mm
from openmm import unit
from openmm import app

from ase import Atoms
from aimnet.calculators import AIMNet2ASE

WATER_NAMES = {"SOL", "HOH", "H2O", "WAT", "TIP3"}

try:
    import yaml
except Exception:
    yaml = None

# ----------------- CLI & Config -----------------

def _coerce_value(s: Any) -> Any:
    # Handle non-string inputs (like booleans from YAML loaded directly)
    if isinstance(s, bool): return s
    if isinstance(s, (int, float)): return s
    
    st = str(s).strip()
    if st.lower() in ("true","yes","on"): return True
    if st.lower() in ("false","no","off"): return False
    try:
        if any(c in st.lower() for c in (".","e")):
            return float(st)
        return int(st)
    except Exception:
        pass
    if "," in st:
        parts = [p.strip() for p in st.split(",") if p.strip()]
        return parts
    try:
        return ast.literal_eval(st)
    except Exception:
        return s

def _add_args(parser: argparse.ArgumentParser) -> None:
    g0 = parser.add_argument_group("Config")
    g0.add_argument("--config", help="YAML/JSON config file. CLI flags override file values.")
    g0.add_argument("--env-file", help=".env/.sh KEY=VALUE (applied after --config, before CLI).")

    g1 = parser.add_argument_group("GROMACS inputs")
    g1.add_argument("--top", help="GROMACS topology (topol.top)")
    g1.add_argument("--gro", help="GROMACS coordinates (conf.gro)")
    g1.add_argument("--include-dir", action="append", default=[], help="Extra include dirs for #include in .top")

    g2 = parser.add_argument_group("AMBER inputs")
    g2.add_argument("--prmtop", help="AMBER prmtop")
    g2.add_argument("--inpcrd", help="AMBER inpcrd/rst7")

    g3 = parser.add_argument_group("Nonbonded/Force Fields")
    g3.add_argument("--nonbonded-cutoff", type=float, default=1.0, help="Nonbonded cutoff distance (nm). Default: 1.0.")
    g3.add_argument("--force-cutoff-method", action="store_true", help="If periodic, use CutoffPeriodic instead of PME.")
    # NEW ARGUMENTS FOR CONSTRAINTS
    g3.add_argument("--constraints", default="HBonds", help="Constraints method: HBonds, AllBonds, HAngles, or None. Default: HBonds")
    g3.add_argument("--rigid-water", default=True, type=_coerce_value, help="Enforce rigid water models (True/False). Default: True")

    parser.add_argument("--ligand-resname", default="MOL", help="Ligand residue name (default: MOL)")
    parser.add_argument("--platform", default="CPU", choices=["CUDA", "OpenCL", "CPU"], help="OpenMM platform (default: CPU)")
    parser.add_argument("--device-index", default=None, help="CUDA/OpenCL device index")
    parser.add_argument("--cpu-threads", type=int, default=None, help="CPU platform thread count")

    parser.add_argument("--segments", type=int, default=20, help="Total number of segments")
    parser.add_argument("--steps-per-seg", type=int, default=5000, help="MD steps per segment during production/warmup")
    parser.add_argument("--verbosity", type=int, default=5000, help="Debug print interval for box/volume (steps); 0 disables debug prints")
    parser.add_argument("--dt", type=float, default=0.002, help="Production time step (ps)")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature (K)")
    parser.add_argument("--friction", type=float, default=1.0, help="Production friction (1/ps)")
    parser.add_argument("--report-interval", type=int, default=1000, help="Console StateDataReporter interval (steps)")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--ewald-error-tol", type=float, default=1e-4, help="PME Ewald error tolerance")
    
    # OUTPUTS
    parser.add_argument("--dcd", default="traj.dcd", help="Binary DCD trajectory filename")
    parser.add_argument("--save-state", default="state.xml", help="Final state XML filename")
    parser.add_argument("--save-minimized", default="minimized.pdb", help="Filename for minimized structure (default: minimized.pdb)")
    parser.add_argument("--restart-state", default="state.xml", help="Optional XML state file to start from (positions/box only)")
    
    parser.add_argument("--initial-minimize", action="store_true", help="Run an initial energy minimization before MD")
    parser.add_argument("--initial-minimize-iter", type=int, default=2000, help="Max iterations for initial minimization")
    parser.add_argument("--warmup-segments", type=int, default=5, help="Number of NVT warmup segments before switching")
    parser.add_argument("--warmup-dt", type=float, default=0.001, help="Warmup time step (ps)")
    parser.add_argument("--warmup-friction", type=float, default=5.0, help="Warmup friction (1/ps)")
    parser.add_argument("--warmup-steps-per-seg", type=int, default=None, help="If set, overrides steps-per-seg during warmup")
    parser.add_argument("--barostat", action="store_true", help="Enable NPT after warmup (requires periodic)")
    parser.add_argument("--pressure", type=float, default=1.0, help="Pressure (bar) for NPT")
    parser.add_argument("--barostat-interval", type=int, default=25, help="Barostat MC interval (steps)")
    parser.add_argument("--switch-to-npt-at-seg", type=int, default=None, help="Segment index to enable NPT (default: warmup-segments)")
    parser.add_argument("--update-every", type=int, default=1, help="Call AIMNet2 every M segments (default 1)")
    parser.add_argument("--alpha", type=float, default=0.2, help="EMA smoothing factor for charges (1=no smoothing)")
    
    # CHARGE CONTROL
    parser.add_argument("--enforce-total-charge", action="store_true", default=True, help="Enforce total system charge to a target")
    parser.add_argument("--target-total-charge", type=float, default=None, help="Override target total charge (e). If not set, uses initial total.")
    parser.add_argument("--q-abs-max", type=float, default=2.0, help="Absolute cap on any atomic charge |q| (e)")
    parser.add_argument("--dq-max", type=float, default=0.05, help="Max per-atom change per update |Δq| (e)")
    
    # NEUTRALITY
    parser.add_argument("--constrain-mol-neutrality", action="store_true", help="Enforce zero net charge per residue (ligand & waters).")
    
    # SPLIT SCALING
    parser.add_argument("--solute-charge-scale", type=float, default=1.0, help="Scale factor for Ligand charges (default 1.0)")
    parser.add_argument("--solvent-charge-scale", type=float, default=1.0, help="Scale factor for Water charges (recommend 1.5 to match TIP3P)")

    parser.add_argument("--post-minimize", action="store_true", default=True, help="Short minimization after each charge update")
    parser.add_argument("--minimize-iter", type=int, default=200, help="Max iterations for post-update minimization")
    parser.add_argument("--reassign-T", action="store_true", default=True, help="Reassign velocities to target T after update")

    parser.add_argument("--xyz-with-charges", default=None, help="Write XYZ with per-atom charges to this file")
    parser.add_argument("--xyz-interval", type=int, default=5000, help="XYZ-with-charges write interval (steps)")
    parser.add_argument("--pdb-with-charges", default=None, help="Write PDB snapshots with charges in B-factor field")
    parser.add_argument("--pdb-interval", type=int, default=25000, help="PDB-with-charges write interval (steps)")

    parser.add_argument("--energies-dat", default="energies.dat", help="Plain-text energetics TSV file")
    parser.add_argument("--energies-interval", type=int, default=1000, help="Energetics logging interval (steps)")

    parser.add_argument("--updates-dat", default=None, help="Per-update diagnostics TSV")
    parser.add_argument("--force-periodic", action="store_true", help="Force periodic if a box is recoverable/provided")
    parser.add_argument("--box", type=str, default=None, help='Manual box "a_nm,b_nm,c_nm,alpha,beta,gamma"')

def _apply_defaults_from_dict(parser: argparse.ArgumentParser, cfg: Dict[str, Any]) -> None:
    cleaned = {k.replace("-","_"): v for k,v in cfg.items()}
    parser.set_defaults(**cleaned)

def parse_with_config(argv: List[str]) -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config"); base.add_argument("--env-file")
    known, _ = base.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="OpenMM segmented MD with AIMNet2 dynamic charges + robust periodic handling")
    _add_args(parser)

    if known.config:
        path = known.config
        if not os.path.exists(path):
            print(f"[ERROR] Config file does not exist: {path}", file=sys.stderr); sys.exit(2)
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r") as fh:
            if ext in (".yaml",".yml"):
                if yaml is None:
                    print("[ERROR] PyYAML not installed; cannot read YAML.", file=sys.stderr); sys.exit(2)
                data = yaml.safe_load(fh)
            else:
                data = json.load(fh)
        if not isinstance(data, dict):
            print("[ERROR] Config must be a mapping/dict.", file=sys.stderr); sys.exit(2)
        _apply_defaults_from_dict(parser, data)

    if known.env_file:
        path = known.env_file
        if not os.path.exists(path):
            print(f"[ERROR] Env file does not exist: {path}", file=sys.stderr); sys.exit(2)
        env_map = {}
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if (not line) or line.startswith("#"): continue
                m = re.match(r'^([A-Za-z_][A-Za-z0-9_\-]*)=(.*)$', line)
                if not m: continue
                key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
                env_map[key.replace("-","_")] = _coerce_value(val)
        _apply_defaults_from_dict(parser, env_map)

    args = parser.parse_args(argv)
    return args

# ----------------- Helpers -----------------

def _get_openmm_constraints(constraints_str: str):
    """Parses constraint string to OpenMM constants."""
    s = str(constraints_str).lower()
    if s == "hbonds": return app.HBonds
    if s == "allbonds": return app.AllBonds
    if s == "hangles": return app.HAngles
    return None # None or "None"

def topology_info(topology: app.Topology, ligand_resname: str):
    atom_to_index: Dict[app.Atom, int] = {}
    atomic_numbers: List[int] = []
    element_symbols: List[str] = []
    
    # For per-molecule neutrality and scaling
    atom_to_res_idx = []
    residue_sizes = []
    is_ligand_atom = []

    for res_i, res in enumerate(topology.residues()):
        n_atoms_in_res = 0
        name = (res.name or "").upper()
        # Simple string match for ligand
        is_lig = (name == ligand_resname.upper())
        
        for atom in res.atoms():
            atom_to_index[atom] = atom.index
            Z = atom.element.atomic_number if atom.element is not None else 0
            atomic_numbers.append(Z)
            element_symbols.append(atom.element.symbol if atom.element is not None else "X")
            atom_to_res_idx.append(res_i)
            is_ligand_atom.append(is_lig)
            n_atoms_in_res += 1
        residue_sizes.append(n_atoms_in_res)

    ligand_atoms: List[int] = []
    waters: List[List[int]] = []
    for res in topology.residues():
        idxs = [atom_to_index[a] for a in res.atoms()]
        name = (res.name or "").upper()
        if name == ligand_resname.upper():
            ligand_atoms.extend(idxs)
        if name in WATER_NAMES:
            waters.append(idxs)
            
    return (np.array(atomic_numbers, dtype=int), 
            element_symbols, 
            ligand_atoms, 
            waters, 
            np.array(atom_to_res_idx, dtype=int), 
            np.array(residue_sizes, dtype=int),
            np.array(is_ligand_atom, dtype=bool))

def _is_periodic_from_box(box_vectors: Optional[Tuple[mm.Vec3, mm.Vec3, mm.Vec3]]) -> bool:
    if box_vectors is None: return False
    try:
        for v in box_vectors:
            if v.norm() <= 0*unit.nanometer: return False
        return True
    except Exception:
        return False

def _parse_box_from_prmtop(prmtop_path: str):
    try:
        with open(prmtop_path, "r") as fh:
            lines = fh.readlines()
    except Exception:
        return None
    try:
        i = next(i for i,l in enumerate(lines) if l.strip().upper().startswith("%FLAG BOX_DIMENSIONS"))
    except StopIteration:
        return None
    toks = []
    j = i + 1
    if j < len(lines) and lines[j].strip().upper().startswith("%FORMAT"):
        j += 1
    while j < len(lines) and not lines[j].strip().startswith("%FLAG"):
        toks += lines[j].split()
        j += 1
    vals = []
    for t in toks:
        try: vals.append(float(t))
        except Exception: pass
    if len(vals) >= 6:
        alpha, beta, gamma, aA, bA, cA = vals[:6]
    elif len(vals) >= 4:
        alpha, aA, bA, cA = vals[:4]
        beta = gamma = 90.0
    else:
        return None
    a_nm, b_nm, c_nm = aA*0.1, bA*0.1, cA*0.1
    return (a_nm, b_nm, c_nm, alpha, beta, gamma)

def _box_vectors_from_lengths_angles(a_nm, b_nm, c_nm, alpha_deg, beta_deg, gamma_deg):
    import math
    alpha = math.radians(alpha_deg); beta = math.radians(beta_deg); gamma = math.radians(gamma_deg)
    ax = a_nm; bx = b_nm*math.cos(gamma); cx = c_nm*math.cos(beta)
    ay = 0.0; by = b_nm*math.sin(gamma)
    if abs(by) < 1e-12:
        cy = 0.0
    else:
        cy = (b_nm*c_nm*math.cos(alpha) - bx*cx)/by
    tmp = c_nm*c_nm - cx*cx - cy*cy
    if tmp < 0 and tmp > -1e-12: tmp = 0.0
    cz = math.sqrt(max(0.0, tmp))
    return (mm.Vec3(ax, 0.0, 0.0)*unit.nanometer,
            mm.Vec3(bx, by, 0.0)*unit.nanometer,
            mm.Vec3(cx, cy, cz)*unit.nanometer)

# Load functions
def load_from_gromacs(top_path: str, gro_path: str, include_dirs: List[str], args: argparse.Namespace):
    gro = app.GromacsGroFile(gro_path)
    top = app.GromacsTopFile(top_path, includeDir=include_dirs)
    box = gro.getPeriodicBoxVectors()
    periodic = _is_periodic_from_box(box)
    cutoff_nm = args.nonbonded_cutoff
    if periodic and not args.force_cutoff_method:
        method = app.PME
    elif periodic:
        method = app.CutoffPeriodic
    else:
        method = app.CutoffNonPeriodic
    
    constraints_obj = _get_openmm_constraints(args.constraints)
    
    kwargs = dict(
        nonbondedMethod=method,
        nonbondedCutoff=cutoff_nm*unit.nanometer,
        constraints=constraints_obj,
        rigidWater=args.rigid_water
    )
    if method == app.PME:
        kwargs['ewaldErrorTolerance'] = args.ewald_error_tol
    system = top.createSystem(**kwargs)
    return top.topology, system, gro.getPositions(), box, periodic

def load_from_amber(prmtop_path: str, inpcrd_path: str, args: argparse.Namespace):
    prmtop = app.AmberPrmtopFile(prmtop_path)
    inp = app.AmberInpcrdFile(inpcrd_path)
    box = inp.boxVectors
    source = "INPCRD"
    if box is None:
        parsed = _parse_box_from_prmtop(prmtop_path)
        if parsed is not None:
            a_nm,b_nm,c_nm,alpha,beta,gamma = parsed
            box = _box_vectors_from_lengths_angles(a_nm,b_nm,c_nm,alpha,beta,gamma)
            source = "PRMTOP(%FLAG BOX_DIMENSIONS)"
    topology = prmtop.topology
    periodic = _is_periodic_from_box(box)
    cutoff_nm = args.nonbonded_cutoff
    if periodic and not args.force_cutoff_method:
        method = app.PME
    elif periodic:
        method = app.CutoffPeriodic
    else:
        method = app.CutoffNonPeriodic
    
    constraints_obj = _get_openmm_constraints(args.constraints)

    kwargs = dict(
        nonbondedMethod=method,
        nonbondedCutoff=cutoff_nm*unit.nanometer,
        constraints=constraints_obj,
        rigidWater=args.rigid_water
    )
    if method == app.PME:
        kwargs['ewaldErrorTolerance'] = args.ewald_error_tol
    system = prmtop.createSystem(**kwargs)
    return topology, system, inp.getPositions(), box, periodic, source

def _rebuild_system_periodic_if_needed(system: mm.System, args: argparse.Namespace, topology: app.Topology, box):
    method = _get_nonbonded_method(system)
    if method is None or not _is_nonperiodic_method(method):
        return system
    cutoff_nm = args.nonbonded_cutoff
    new_method = app.CutoffPeriodic if args.force_cutoff_method else app.PME
    print(f"[BOX] System rebuilt with {'CutoffPeriodic' if args.force_cutoff_method else 'PME'} and cutoff {cutoff_nm} nm.")
    
    constraints_obj = _get_openmm_constraints(args.constraints)

    kwargs = dict(
        nonbondedMethod=new_method,
        nonbondedCutoff=cutoff_nm*unit.nanometer,
        constraints=constraints_obj,
        rigidWater=args.rigid_water
    )
    if new_method == app.PME:
        kwargs['ewaldErrorTolerance'] = args.ewald_error_tol

    if getattr(args, "prmtop", None):
        prmtop_reload = app.AmberPrmtopFile(args.prmtop)
        system = prmtop_reload.createSystem(**kwargs)
    elif getattr(args, "top", None):
        top_reload = app.GromacsTopFile(args.top, includeDir=getattr(args, "include_dir", []))
        system = top_reload.createSystem(**kwargs)
    else:
        raise RuntimeError("Periodic box detected/forced but no topology source to rebuild the System.")
    if box is not None:
        system.setDefaultPeriodicBoxVectors(*box)
        try: topology.setPeriodicBoxVectors(box)
        except Exception: pass
    return system

def get_nonbonded_force(system: mm.System) -> mm.NonbondedForce:
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce): return f
    raise RuntimeError("No NonbondedForce found.")

def _get_nonbonded_method(system: mm.System):
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce): return f.getNonbondedMethod()
    return None

def _is_nonperiodic_method(method: int) -> bool:
    return method in (mm.NonbondedForce.NoCutoff, mm.NonbondedForce.CutoffNonPeriodic)

def get_current_charges(system: mm.System) -> np.ndarray:
    nb = get_nonbonded_force(system)
    q = np.zeros(nb.getNumParticles(), dtype=float)
    for i in range(nb.getNumParticles()):
        charge, _, _ = nb.getParticleParameters(i)
        q[i] = charge.value_in_unit(unit.elementary_charge)
    return q

def aimnet2_predict_charges(numbers: np.ndarray, positions_nm: np.ndarray, box_vectors_nm=None) -> np.ndarray:
    pos_ang = positions_nm * 10.0
    if box_vectors_nm is not None:
        if hasattr(box_vectors_nm, 'value_in_unit'):
             cell_nm = box_vectors_nm.value_in_unit(unit.nanometer)
        else:
             cell_nm = box_vectors_nm
        v1 = [cell_nm[0][0], cell_nm[0][1], cell_nm[0][2]]
        v2 = [cell_nm[1][0], cell_nm[1][1], cell_nm[1][2]]
        v3 = [cell_nm[2][0], cell_nm[2][1], cell_nm[2][2]]
        cell_ang = np.array([v1, v2, v3]) * 10.0
        atoms = Atoms(numbers=numbers.tolist(), positions=pos_ang, cell=cell_ang, pbc=True)
    else:
        atoms = Atoms(numbers=numbers.tolist(), positions=pos_ang)
    
    atoms.calc = AIMNet2ASE("aimnet2")
    try:
        q = atoms.get_charges()
        if q is None: raise RuntimeError("atoms.get_charges() returned None")
        q = np.asarray(q, dtype=float)
    except Exception:
        q = atoms.calc.get_property("charges", atoms)
        q = np.asarray(q, dtype=float)
    if not np.all(np.isfinite(q)):
        raise RuntimeError("AIMNet2 returned non-finite charges.")
    if q.shape[0] != numbers.shape[0]:
        raise RuntimeError(f"AIMNet2 returned {q.shape[0]} charges")
    return q

def update_nonbonded_charges_in_context(system: mm.System, context: mm.Context, new_charges: np.ndarray):
    nb = get_nonbonded_force(system)
    for i in range(nb.getNumParticles()):
        _, sigma, epsilon = nb.getParticleParameters(i)
        nb.setParticleParameters(i, new_charges[i]*unit.elementary_charge, sigma, epsilon)
    nb.updateParametersInContext(context)

def write_xyz_with_charges(filename: str, element_symbols: list, positions_nm: np.ndarray, charges: np.ndarray,
                           step: int, time_ps: float, box_vectors=None, append=True):
    nat = len(element_symbols)
    pos_ang = positions_nm * 10.0
    mode = "a" if append and os.path.exists(filename) else "w"
    def _len_nm_vec3(v):
        def _as_nm(val):
            try: return float(val.value_in_unit(unit.nanometer))
            except Exception: return float(val)
        return ( _as_nm(v.x)**2 + _as_nm(v.y)**2 + _as_nm(v.z)**2 ) ** 0.5
    with open(filename, mode) as fh:
        fh.write(f"{nat}\n")
        if box_vectors is not None:
            a,b,c = box_vectors
            la, lb, lc = _len_nm_vec3(a)*10.0, _len_nm_vec3(b)*10.0, _len_nm_vec3(c)*10.0
            fh.write(f" step={step} time_ps={time_ps:.6f} box=[{la:.3f},{lb:.3f},{lc:.3f}] columns: El x(Å) y(Å) z(Å) q(e)\n")
        else:
            fh.write(f" step={step} time_ps={time_ps:.6f} columns: El x(Å) y(Å) z(Å) q(e)\n")
        for el,(x,y,z),q in zip(element_symbols, pos_ang, charges):
            fh.write(f"{el:2s} {x:12.6f} {y:12.6f} {z:12.6f} {q: .6f}\n")

def write_pdb_with_charges(filename: str, topology: app.Topology, positions_nm: np.ndarray, charges: np.ndarray, append=True):
    pos_ang = positions_nm * 10.0
    mode = "a" if append and os.path.exists(filename) else "w"
    with open(filename, mode) as fh:
        fh.write("MODEL\n")
        atom_index = 1
        for res in topology.residues():
            for atom in res.atoms():
                i = atom.index
                x,y,z = pos_ang[i]
                name = (atom.name or "X")[:4].rjust(4)
                resn = (res.name or "RES")[:3]
                chainid = (res.chain.id or "A")[:1]
                bf = charges[i]
                fh.write(f"ATOM  {atom_index:5d} {name} {resn:3s} {chainid:1s}{res.id:>4s}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bf:6.2f}\n")
                atom_index += 1
        fh.write("ENDMDL\n")

class TSVEnergyReporter:
    def __init__(self, file, reportInterval, periodic, system, charge_getter, write_header=True, verbosity=None):
        self._file = file
        self._interval = int(reportInterval)
        self._periodic = bool(periodic)
        self._system = system
        self._charge_getter = charge_getter
        self._write_header = write_header
        self._verbosity = int(verbosity) if (verbosity and int(verbosity)>0) else None
        self._total_mass_amu = 0.0
        for i in range(system.getNumParticles()):
            self._total_mass_amu += system.getParticleMass(i).value_in_unit(unit.amu)

    def describeNextReport(self, simulation):
        return (self._interval, False, False, False, True, self._periodic)

    def report(self, simulation, state):
        new_file = not os.path.exists(self._file) or os.path.getsize(self._file) == 0
        with open(self._file, "a") as f:
            if self._write_header and new_file:
                f.write("# step\ttime_ps\tPE_kJmol\tKE_kJmol\tTE_kJmol\tT_K\tV_nm3\tdensity_gmL\tcharge_sum_e\n")
            step = simulation.currentStep
            time_ps = state.getTime().value_in_unit(unit.picoseconds)
            PE = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            KE = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            TE = PE + KE
            try:
                N = self._system.getNumParticles()
                ncon = self._system.getNumConstraints()
                has_cmm = any(isinstance(fr, mm.CMMotionRemover) for fr in self._system.getForces())
                dof = max(1, 3*N - ncon - (3 if has_cmm else 0))
                kB = (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
                T = 2.0*KE/(kB*dof)
                T_str = f"{T:.3f}"
            except Exception: T_str = "NA"

            if self._periodic:
                box_vectors = state.getPeriodicBoxVectors()
                def _len(v): return np.sqrt(v[0].value_in_unit(unit.nanometer)**2 + v[1].value_in_unit(unit.nanometer)**2 + v[2].value_in_unit(unit.nanometer)**2)
                a_nm, b_nm, c_nm = _len(box_vectors[0]), _len(box_vectors[1]), _len(box_vectors[2])
                
                a_vec = np.array([box_vectors[0][0].value_in_unit(unit.nanometer), box_vectors[0][1].value_in_unit(unit.nanometer), box_vectors[0][2].value_in_unit(unit.nanometer)])
                b_vec = np.array([box_vectors[1][0].value_in_unit(unit.nanometer), box_vectors[1][1].value_in_unit(unit.nanometer), box_vectors[1][2].value_in_unit(unit.nanometer)])
                c_vec = np.array([box_vectors[2][0].value_in_unit(unit.nanometer), box_vectors[2][1].value_in_unit(unit.nanometer), box_vectors[2][2].value_in_unit(unit.nanometer)])
                volume_nm3 = abs(np.dot(a_vec, np.cross(b_vec, c_vec)))
                V_str = f"{volume_nm3:.6f}"
                density_str = f"{self._total_mass_amu / 6.02214076e23 / (volume_nm3 * 1e-21) if volume_nm3 > 0 else 0:.6f}"
                
                if self._verbosity and step % self._verbosity == 0:
                    print(f"[DEBUG] Step {step}: V={volume_nm3:.3f} nm3, Rho={density_str} g/mL")
            else:
                V_str = "NA"; density_str = "NA"

            qsum = float(np.sum(self._charge_getter()))
            f.write(f"{step}\t{time_ps:.6f}\t{PE:.6f}\t{KE:.6f}\t{TE:.6f}\t{T_str}\t{V_str}\t{density_str}\t{qsum:.6f}\n")

# ----------------- Main -----------------

def setup_reporters(sim, args, filenames, periodic, system):
    sim.reporters.clear()
    sim.reporters.append(app.DCDReporter(filenames['dcd'], args.report_interval))
    sim.reporters.append(app.StateDataReporter(sys.stdout, args.report_interval, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True, separator="\t"))
    if filenames['energies']:
        sim.reporters.append(TSVEnergyReporter(filenames['energies'], args.energies_interval, periodic, system, lambda: get_current_charges(system), verbosity=args.verbosity))

def enable_barostat(sim, system, topology, pressure_bar, temperature_K, interval, platform, props, integrator, args, periodic):
    state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True)
    state_xml = mm.XmlSerializer.serialize(state)
    current_step = sim.currentStep
    
    # We must rebuild sim to add Force
    system.addForce(mm.MonteCarloBarostat(pressure_bar*unit.bar, temperature_K*unit.kelvin, interval))
    
    # Recreate Integrator/Sim
    # Note: We must create a NEW integrator instance
    new_integrator = mm.LangevinMiddleIntegrator(temperature_K*unit.kelvin, integrator.getFriction(), integrator.getStepSize())
    new_integrator.setRandomNumberSeed(integrator.getRandomNumberSeed())
    
    new_sim = app.Simulation(topology, system, new_integrator, platform, props)
    new_sim.context.setState(mm.XmlSerializer.deserialize(state_xml))
    new_sim.currentStep = current_step
    
    return new_sim

def _parse_manual_box(box_str: str):
    parts = [float(x) for x in box_str.replace(",", " ").split() if x.strip()]
    if len(parts) != 6: raise ValueError("--box requires 6 values")
    return _box_vectors_from_lengths_angles(*parts)

def main(argv=None):
    args = parse_with_config(sys.argv[1:] if argv is None else argv)
    manual_box = _parse_manual_box(args.box) if args.box else None

    use_amber = (args.prmtop is not None or args.inpcrd is not None)
    use_gmx = (args.top is not None or args.gro is not None)
    if use_amber == use_gmx:
        print("[ERROR] Specify either AMBER or GROMACS inputs.", file=sys.stderr); sys.exit(2)

    if use_amber:
        topology, system, positions, box, periodic, source = load_from_amber(args.prmtop, args.inpcrd, args)
        print(f"[BOX] Periodic: {periodic} from {source}" + (" (manual box provided)" if manual_box else ""))
    else:
        topology, system, positions, box, periodic = load_from_gromacs(args.top, args.gro, args.include_dir, args)
        print("[BOX] Periodic: True (GROMACS)" + (" (manual box provided)" if manual_box else ""))

    if manual_box is not None:
        box = manual_box; periodic = True
        print("[BOX] Using manual box.")
    if not periodic and args.force_periodic and box is not None:
        periodic = True
        print("[BOX] Forcing periodic.")

    if args.restart_state:
        print(f"[RESTART] Loading {args.restart_state}")
        with open(args.restart_state, "r") as fh: restart_state = mm.XmlSerializer.deserialize(fh.read())
        positions = restart_state.getPositions()
        st_box = restart_state.getPeriodicBoxVectors()
        if st_box:
            box = st_box; periodic = True
            print("[RESTART] Box loaded from state.")

    if periodic and box is not None:
        system.setDefaultPeriodicBoxVectors(*box)
        try: topology.setPeriodicBoxVectors(box)
        except Exception: pass
        system = _rebuild_system_periodic_if_needed(system, args, topology, box)
    elif args.barostat:
        print("[WARN] Nonperiodic system: disabling barostat."); args.barostat = False

    integrator = mm.LangevinMiddleIntegrator(args.temperature*unit.kelvin, args.friction/unit.picosecond, args.dt*unit.picoseconds)
    integrator.setRandomNumberSeed(args.seed)
    
    platform = mm.Platform.getPlatformByName(args.platform)
    props = {}
    if args.platform == "CPU" and args.cpu_threads: props["Threads"] = str(args.cpu_threads)
    if args.platform in ("CUDA","OpenCL") and args.device_index:
        key = "CudaDeviceIndex" if args.platform=="CUDA" else "OpenCLDeviceIndex"
        props[key] = str(args.device_index)
        if args.platform=="CUDA": props["CudaPrecision"] = "mixed"

    sim = app.Simulation(topology, system, integrator, platform, props)
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(args.temperature*unit.kelvin, args.seed)

    # Initial minimization
    if args.initial_minimize:
        print(f"[INIT] Minimizing (maxIter={args.initial_minimize_iter})...")
        mm.LocalEnergyMinimizer.minimize(sim.context, tolerance=10*unit.kilojoule_per_mole/unit.nanometer, maxIterations=args.initial_minimize_iter)
        print("[INIT] Done.")
        if args.save_minimized:
            print(f"[INIT] Saving minimized structure to {args.save_minimized}")
            state_min = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
            # Sync topology box with current state box to ensure consistency/validity
            if periodic:
                topology.setPeriodicBoxVectors(state_min.getPeriodicBoxVectors())
            try:
                with open(args.save_minimized, 'w') as f:
                    app.PDBFile.writeFile(topology, state_min.getPositions(), f)
            except Exception as e:
                print(f"[WARN] Failed to write minimized PDB: {e}")
        
        sim.context.setVelocitiesToTemperature(args.temperature*unit.kelvin, args.seed)

    if periodic and box is not None:
        a, b, c = box
        sim.context.setPeriodicBoxVectors(a, b, c)

    # Filename Handling
    files = {
        'dcd': args.dcd,
        'energies': args.energies_dat,
        'updates': args.updates_dat,
        'xyz': args.xyz_with_charges,
        'pdb': args.pdb_with_charges
    }
    warmup_files = {k: f"warmup_{v}" if v else None for k,v in files.items()}
    
    current_files = warmup_files if args.warmup_segments > 0 else files
    setup_reporters(sim, args, current_files, periodic, system)

    # Topology analysis
    numbers, element_symbols, ligand_idx, waters, atom_to_res_idx, residue_sizes, is_ligand_atom = topology_info(topology, args.ligand_resname)
    
    q_initial_ff = get_current_charges(system)
    q_prev = q_initial_ff.copy()
    target_total = float(np.sum(q_prev)) if args.target_total_charge is None else float(args.target_total_charge)
    print(f"[INIT] Target total charge: {target_total:+.6f} e")

    switch_seg = args.warmup_segments if args.switch_to_npt_at_seg is None else args.switch_to_npt_at_seg
    npt_enabled = False
    last_xyz_step = -1e9

    for seg in range(args.segments):
        if seg < args.warmup_segments:
            integrator.setStepSize(args.warmup_dt*unit.picoseconds)
            integrator.setFriction(args.warmup_friction/unit.picosecond)
            steps_now = args.warmup_steps_per_seg if args.warmup_steps_per_seg else args.steps_per_seg
        else:
            integrator.setStepSize(args.dt*unit.picoseconds)
            integrator.setFriction(args.friction/unit.picosecond)
            steps_now = args.steps_per_seg

        # Transition Logic
        if seg == args.warmup_segments and args.warmup_segments > 0:
             print("[TRANSITION] Warmup complete. Switching to production output files.")
             current_files = files
             # Note: sim must be re-wrapped if we changed reporters on the fly?
             # Actually setup_reporters clears sim.reporters.
             setup_reporters(sim, args, current_files, periodic, system)

        # Barostat logic
        if args.barostat and (not npt_enabled) and periodic and (seg >= switch_seg):
            print(f"[SWITCH] NPT enabled at seg {seg}")
            # Note: enable_barostat returns a NEW sim object. We must re-attach reporters!
            sim = enable_barostat(sim, system, topology, args.pressure, args.temperature, args.barostat_interval, platform, props, integrator, args, periodic)
            setup_reporters(sim, args, current_files, periodic, system)
            npt_enabled = True
        
        sim.step(steps_now)

        if (seg % args.update_every) != 0: continue
        if seg < args.warmup_segments:
            print(f"[SEG {seg}] Warmup (fixed charges).")
            continue

        # Charge Update
        state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box_vectors = state.getPeriodicBoxVectors()
        
        # 1. Predict (ALL atoms)
        q_aimnet = aimnet2_predict_charges(np.array(numbers), pos_nm, box_vectors_nm=box_vectors)

        # 2. Split Scaling
        scale_arr = np.ones_like(q_aimnet)
        scale_arr[is_ligand_atom] = args.solute_charge_scale
        scale_arr[~is_ligand_atom] = args.solvent_charge_scale
        q_new = q_aimnet * scale_arr

        # 4. Guardrails
        if args.q_abs_max > 0:
            np.clip(q_new, -args.q_abs_max, args.q_abs_max, out=q_new)
        
        # 5. Per-Molecule Neutrality
        if args.constrain_mol_neutrality:
            res_charges = np.zeros(len(residue_sizes))
            np.add.at(res_charges, atom_to_res_idx, q_new)
            res_corrections = -res_charges / residue_sizes
            q_new += res_corrections[atom_to_res_idx]

        # 6. Delta Limit
        if args.dq_max > 0 and q_prev is not None:
            dq = q_new - q_prev
            np.clip(dq, -args.dq_max, args.dq_max, out=dq)
            q_new = q_prev + dq

        # 7. EMA
        a = float(args.alpha)
        if a < 1.0 and q_prev is not None:
            q_to_set = a*q_new + (1.0 - a)*q_prev
        else:
            q_to_set = q_new.copy()

        # 8. Total Charge Enforce
        if args.enforce_total_charge:
            shift = (target_total - float(np.sum(q_to_set))) / q_to_set.size
            q_to_set += shift
        else:
            shift = 0.0

        old = get_current_charges(system)
        pe_before = sim.context.getState(getEnergy=True).getPotentialEnergy()
        update_nonbonded_charges_in_context(system, sim.context, q_to_set)
        
        if args.post_minimize:
            mm.LocalEnergyMinimizer.minimize(sim.context, tolerance=10.0*unit.kilojoule_per_mole/unit.nanometer, maxIterations=args.minimize_iter)
        if args.reassign_T:
            sim.context.setVelocitiesToTemperature(args.temperature*unit.kelvin, args.seed + seg + 1)
        
        dPE = (sim.context.getState(getEnergy=True).getPotentialEnergy() - pe_before).value_in_unit(unit.kilojoule_per_mole)
        print(f"[SEG {seg}] Σq={np.sum(q_to_set):.6f}, scale_solv={args.solvent_charge_scale}, dPE={dPE:.3f}")
        
        if current_files['updates']:
            delta = q_to_set - old
            time_ps = state.getTime().value_in_unit(unit.picoseconds)
            qsum_total = float(np.sum(q_to_set))
            # RESTORED 10-COLUMN FORMAT for compatibility
            with open(current_files['updates'], "a") as fh:
                fh.write(f"{seg}\t{sim.currentStep}\t{time_ps:.6f}\t{dPE:.6f}\t{shift:.6e}\t"
                         f"{delta.mean():.6e}\t{delta.std():.6e}\t{delta.min():.6e}\t{delta.max():.6e}\t"
                         f"{qsum_total:.6f}\n")
        
        q_prev = get_current_charges(system)
        
        step_now = sim.currentStep
        if current_files['xyz'] and (step_now - last_xyz_step) >= args.xyz_interval:
            write_xyz_with_charges(current_files['xyz'], element_symbols, pos_nm, q_prev, step_now, state.getTime().value_in_unit(unit.picoseconds), box_vectors)
            last_xyz_step = step_now

    print(f"Done. State saved to {args.save_state}")

if __name__ == "__main__":
    main()
