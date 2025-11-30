#!/bin/bash
workdir=$(pwd)
ligand=MOL

#antechamber -i MOL.esp -fi gesp -o "$ligand".prepin -fo prepi -c resp -s 2 -rn MOL -at gaff2 -nc 0 -pf yes
antechamber -i MOL.mol2 -fi mol2 -o "$ligand".prepin -fo prepi -c abcg2 -s 2 -rn MOL -at gaff2 -nc 0 -pf yes
parmchk2 -i "$ligand".prepin -f prepi -o "$ligand".frcmod

tleap -f - <<EOF
source leaprc.gaff2
source leaprc.water.tip3p
loadamberparams $ligand.frcmod
loadamberprep $ligand.prepin
solvatebox MOL TIP3PBOX 10.0
saveamberparm MOL MOL_solv.prmtop MOL_solv.inpcrd
quit
EOF

### ---- 2) Convert AMBER -> GROMACS using ParmEd (inline python), with renaming ----
#python - <<'PY'
#from parmed import load_file
#
#amber = load_file('MOL_solv.prmtop', 'MOL_solv.inpcrd')
#
## Normalize residue names:
## Waters: WAT/HOH -> SOL
## Ions:   Na+, Na, NA+ -> NA     ;  Cl-, Cl, CL- -> CL
#for res in amber.residues:
#    rn = res.name.strip()
#    if rn in ('WAT', 'HOH'):
#        res.name = 'SOL'
#    elif rn in ('Na+', 'NA+', 'Na', 'NA'):
#        res.name = 'NA'
#    elif rn in ('Cl-', 'CL-', 'Cl', 'CL'):
#        res.name = 'CL'
#    # keep ZN as 'ZN' (divalent metal cofactor), do not rename
#
#amber.save('MOL_solv.top', format='gromacs')
#amber.save('MOL_solv.gro')
#PY
#


