from rdkit import Chem
m = Chem.SDMolSupplier("./8cpa_ligand.sdf")
smi = Chem.MolToSmiles(m[0])
print(smi)
mols = [Chem.MolToSmiles(mol) for mol in m if mol]
print(mols)