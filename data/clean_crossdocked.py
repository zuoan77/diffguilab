import os
import sys
import shutil
import gzip
import pickle
import argparse
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

TYPES_FILENAME = 'types/it2_tt_v1.1_completeset_train0.types'  # 'types/it2_tt_completeset_train0.types'

if __name__ == '__main__':
    # Useage: python clean_crossdocked.py --source Crossdock2020_v1.1 --dest crossdocked_v1.1_rmsd1.0 --rmsd_thr 1.0
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./data/CrossDocked2020')
    parser.add_argument('--dest', type=str, required=True)
    parser.add_argument('--rmsd_thr', type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=False)
    types_path = os.path.join(args.source, TYPES_FILENAME)

    index = []
    with open(types_path, 'r') as f:
        for ln in tqdm(f.readlines()):
            _, _, rmsd, protein_fn, ligand_fn, _ = ln.split()
            rmsd = float(rmsd)
            if rmsd > args.rmsd_thr: 
                continue

            ligand_id = int(ligand_fn[ligand_fn.rfind('_')+1:ligand_fn.rfind('.')])

            protein_fn = protein_fn[:protein_fn.rfind('_')] + '.pdb'
            # For CrossDocked v1.0
            ligand_raw_fn = ligand_fn[:ligand_fn.rfind('_')] + '.sdf.gz'
            protein_path = os.path.join(args.source, protein_fn)
            ligand_raw_path = os.path.join(args.source, ligand_raw_fn)
            if not (os.path.exists(protein_path) and os.path.exists(ligand_raw_path)):
                continue

            # For CrossDocked v1.0
            with gzip.open(ligand_raw_path, 'rt') as f:
                ligand_sdf = f.read().split('$$$$\n')[ligand_id]
            ligand_save_fn = ligand_fn[:ligand_fn.rfind('.')] + '.sdf'  # include ligand id

            protein_dest = os.path.join(args.dest, protein_fn)
            ligand_dest = os.path.join(args.dest, ligand_save_fn)
            os.makedirs(os.path.dirname(protein_dest), exist_ok=True)
            os.makedirs(os.path.dirname(ligand_dest), exist_ok=True)
            shutil.copyfile(protein_path, protein_dest)
            with open(ligand_dest, 'w') as f:
                f.write(ligand_sdf)
            index.append((protein_fn, ligand_save_fn, rmsd))

    index_path = os.path.join(args.dest, 'index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump(index, f)

    print(f'Done processing {len(index)} protein-ligand pairs in total.\n Processed files in {args.dest}.')
