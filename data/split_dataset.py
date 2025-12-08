import os
import torch
import random
import argparse
from tqdm import tqdm
from torch.utils.data import Subset
from utils.dataset import ProteinLigandDataset


def get_unique_pockets(dataset, raw_id, used_pdb, num, args):
    unique_id = []
    pdb_visited = set()
    for idx in tqdm(raw_id, desc="Filter"):
        pdb_name = os.path.basename(dataset[idx].ligand_filename)[:4]
        if pdb_name not in used_pdb and pdb_name not in pdb_visited:
            unique_id.append(idx)
            pdb_visited.add(pdb_name)
    print(f"Number of pairs: {len(unique_id)}.")
    print(f"Number of pdbs: {len(pdb_visited)}.")
    random.Random(args.seed).shuffle(unique_id)
    unique_id = unique_id[:num]
    print(f"Number of selected pockets: {len(unique_id)}.")
    pdb_visited = pdb_visited.union(used_pdb)
    return unique_id, pdb_visited

def main(args):
    dataset = ProteinLigandDataset(args.path)
    print("Load dataset successfully!")
    if args.fixed_split is not None:
        fixed_split = torch.load(args.fixed_split)
        print("Load fixed split successfully!")
        name_id = {}
        for idx, data in enumerate(tqdm(dataset, desc="Indexing")):
            name_id[data.protein_filename + data.ligand_filename] = idx
        
        selected_ids = {"train": [], "test": []}
        for split in ["train", "test"]:
            print(f"Selecting {split} split.")
            for sp in fixed_split[split]:
                if (sp[0] + sp[1]) in name_id:
                    selected_ids[split].append(name_id[sp[0] + sp[1]])
                else:
                    print(f"Data with pocket {sp[0]} and ligand {sp[1]} does not exist!")
        train_id, val_id, test_id = selected_ids["train"], [], selected_ids["test"]

    else:
        allowed_lig_elements = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
        elements = {i: set() for i in range(90)}
        for i, data in enumerate(tqdm(dataset, desc="Filter")):
            for e in data.ligand_element:
                elements[e.item()].add(i)
        all_id = set(range(len(dataset)))
        blocked_id = set().union(
            *[elements[i] for i in elements.keys() if i not in allowed_lig_elements]
        )
        allowed_id = list(all_id - blocked_id)
        random.Random(args.seed).shuffle(allowed_id)
        print(f"Allowed: {len(allowed_id)}")

        train_id = allowed_id[:args.train]
        train_set = Subset(dataset, indices=train_id)
        train_pdb = {os.path.basename(data.ligand_filename)[:4] for data in tqdm(train_set)}
        print(f"Train pdb: {train_pdb}")

        if args.val_num == -1:
            val_id = allowed_id[args.train: args.train + args.val]
            used_pdb = train_pdb
        else:
            raw_val_id = allowed_id[args.train: args.train + args.val]
            val_id, used_pdb = get_unique_pockets(
                dataset, raw_val_id, train_pdb, args.val_num, args
            )

        if args.test_num == -1:
            test_id = allowed_id[args.train + args.val: args.train + args.val + args.test]
        else:
            raw_test_id = allowed_id[args.train + args.val: args.train + args.val + args.test]
            test_id, used_pdb = get_unique_pockets(
                dataset, raw_test_id, used_pdb, args.test_num, args
            )
    
    torch.save({"train": train_id, "val": val_id, "test": test_id}, args.desti)
    print(f"Train set {len(train_id)}, Validation set {len(val_id)}, Test set {len(test_id)}")


if __name__ == "__main__":
    # Useage: python split_dataset.py --path ./PDBbind_v2020_pocket10 --desti ./PDBbind_pocket10_split.pt --train 17327 --val 1825 --test 100
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./PDBbind_v2020_pocket10")
    parser.add_argument("--desti", type=str, default="./PDBbind_pocket10_split.pt")
    parser.add_argument("--fixed_split", type=str, default=None)
    parser.add_argument("--train", type=int, default=15000)
    parser.add_argument("--val", type=int, default=2000)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--val_num", type=int, default=-1)
    parser.add_argument("--test_num", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    main(args)
