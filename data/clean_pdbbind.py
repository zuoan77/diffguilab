import os
import sys
import pickle
import argparse

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


KMAP = {'Ki': 1, 'Kd': 2, 'IC50': 3}

def main(args):
    index = []
    index_path = os.path.join(args.source, "index/INDEX_general_PL_data.2020")
    with open(index_path, 'r') as fr:
        lines = fr.readlines()
    for line in tqdm(lines):
        if line.startswith('#'):
            continue
        else:
            pdbid, res, year, pka, kv = line.split('//')[0].strip().split()
            kind = [v for k, v in KMAP.items() if k in kv]
            assert len(kind) == 1

            try:
                protein_fn = pdbid + "/" + pdbid + "_protein.pdb"
                ligand_fn = pdbid + "/" + pdbid + "_ligand.sdf"
                ligand = "PDBbind_v2020/" + ligand_fn
                if os.path.getsize(ligand) == 0:
                    os.system('obabel ' + ligand.split('.')[0] + '.mol2 -O' + ligand)
                mol = Chem.SDMolSupplier(ligand)[0]
                logp = round(Descriptors.MolLogP(mol), 4)
                tpsa = round(rdMolDescriptors.CalcTPSA(mol), 4)
                sascore = round(sascorer.calculateScore(mol), 4)
                qed_score = round(QED.qed(mol), 4)
                index.append((protein_fn, ligand_fn, pdbid, logp, tpsa, sascore, qed_score, pka, kind[0]))
            except Exception as e:
                print(pdbid, str(e))
                continue
    
    new_index_path = os.path.join(args.source, "index.pkl")
    with open(new_index_path, "wb") as f:
        pickle.dump(index, f)
    print(f"Processing {len(index)} protein-ligand pais. Processed data is stored in {args.source}")


if __name__ == "__main__":
    # Useage: python clean_pdbbind.py --source ./PDBbind_v2020
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./PDBbind_v2020")
    args = parser.parse_args()

    main(args)
