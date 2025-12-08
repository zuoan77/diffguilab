import os
import pickle
import shutil
import argparse
import multiprocessing
from tqdm import tqdm
from functools import partial
from utils.data import PDBProtein, parse_drug3d_mol


def load_item(item, path):
    protein_path = os.path.join(path, item[0])
    ligand_path = os.path.join(path, item[1])
    with open(protein_path, "r") as f:
        protein_block = f.read()
    with open(ligand_path, "r") as f:
        ligand_block = f.read()
    return protein_block, ligand_block

def process_item(item, args):
    try:
        print(item)
        protein_block, _ = load_item(item, args.source)
        protein = PDBProtein(protein_block)
        ligand = parse_drug3d_mol(os.path.join(args.source, item[1]))

        pdb_pocket_block = protein.residues_to_pdb_block(
            protein.query_residues_ligand(ligand, args.radius)
        )
        ligand_desti = os.path.join(args.desti, item[1])
        pdbid = item[0].split("/")[0]
        pocket_fn = pdbid + "/" + pdbid + f"_pocket_{args.radius}.pdb"
        pocket_desti = os.path.join(args.desti, pocket_fn)
        os.makedirs(os.path.dirname(ligand_desti), exist_ok=True)
        shutil.copy(os.path.join(args.source, item[1]), ligand_desti)

        with open(pocket_desti, "w") as f:
            f.write(pdb_pocket_block)
        return pocket_fn, item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8]
    except Exception:
        print("Error occurred!", item)
        return None, item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7], item[8]

def main(args):
    os.makedirs(args.desti, exist_ok=False)
    with open(os.path.join(args.source, "index.pkl"), "rb") as f:
        index = pickle.load(f)

    pool = multiprocessing.Pool(args.num_workers)
    index_pocket = []
    for item_pocket in tqdm(pool.imap_unordered(partial(process_item, args=args), index), total=len(index)):
        index_pocket.append(item_pocket)
    pool.close()

    with open(os.path.join(args.desti, "index.pkl"), "wb") as f:
        pickle.dump(index_pocket, f)
    print(f"Extracting pockets of {len(index)} protein-ligand pairs.")


if __name__ == "__main__":
    # Useage: python extract_pockets.py --source ./PDBbind_v2020 --desti ./PDBbind_v2020_pocket10
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./PDBbind_v2020")
    parser.add_argument("--desti", type=str, required=True, default="./PDBbind_v2020_pocket10")
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    main(args)
