from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
smi = '[N-]=[N+]=O'
mol = Chem.MolFromSmiles(smi)
Draw.MolToFile(mol,'graph10.png')
print('*****')