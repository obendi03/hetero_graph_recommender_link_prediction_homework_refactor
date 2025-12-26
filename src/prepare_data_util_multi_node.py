from pathlib import Path

import torch
import numpy as np
from torch_geometric.data import HeteroData
from collections import defaultdict

def load_user_item_triplets(path_to_file):
    print("load_user_item_triplets started ")

    user_to_id = {}
    item_to_id = {}
    relation_to_id = {}

    triplets = []
    most_popular_100_train_items = ['1658', '5175', '2980', '94', '3053', '3565', '334', '4404', '3584', '4395', '2740', '3059', '5326', '7447', '5435', '4393', '476', '4398', '7380', '3568', '4458', '5323', '3575', '2603', '5177', '1653', '7016', '5809', '5404', '3332', '6494', '415', '4935', '5096', '82', '2922', '2482', '231', '2450', '2989', '7928', '433', '5403', '3995', '628', '6385', '4630', '6965', '3756', '5118', '4936', '3439', '5233', '5405', '6878', '2839', '5393', '3057', '119', '3812', '5564', '4627', '5799', '2904', '2316', '175', '5103', '4038', '2287', '3889', '3128', '5355', '7426', '4310', '4405', '3447', '2467', '5511', '249', '2522', '4926', '5538', '159', '7437', '2367', '7280', '6820', '5534', '4360', '4970', '6800', '1610', '661', '5574', '4504', '4302', '4096', '3989', '3088', '4636']
    map_original_most_popular_to_new = {}
    with open(path_to_file, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    written = False
    # Minden sor: user, relation, item
    for line in lines:
        user, rel, item = line.split("\t")

        if user not in user_to_id:
            user_to_id[user] = len(user_to_id)
        if item not in item_to_id:
            item_to_id[item] = len(item_to_id)
        if rel not in relation_to_id:
            relation_to_id[rel] = len(relation_to_id)

        if item=="1658" and not written:
            print(item_to_id[item])
            written = True

        triplets.append((user, rel, item))


    #region get item id of most popular oines
    for original_item_id in most_popular_100_train_items:
        map_original_most_popular_to_new[original_item_id] = item_to_id[original_item_id]
    #endregion
    print(map_original_most_popular_to_new)

    #region gett mos popular items
    from collections import Counter

    # Leggyakoribb item (tail) meghatározása
    # item_counts: item → gyakoriság

    item_counts = Counter([item for (_, _, item) in triplets])

    # top 100 leggyakoribb
    top_100 = item_counts.most_common(100)

    item_ids = [item for item, _ in top_100]
    print("top 100",item_ids)

    print("🔥 Top 100 leggyakoribb item rangsora:\n")

    for rank, (item, freq) in enumerate(top_100, start=1):
        item_id = item_to_id[item]
        print(f"#{rank:3d} | Item: {item} | ID: {item_id} | Előfordulás: {freq}")

    print("")
    #endregion
    num_users = len(user_to_id)
    num_items = len(item_to_id)
    num_relations = len(relation_to_id)

    print(f"📊 Users: {num_users}, Items: {num_items}, Relations: {num_relations}")

    # --- Degree-alapú feature ---
    user_features = np.zeros((num_users, num_relations), dtype=float)
    item_features = np.zeros((num_items, num_relations), dtype=float)

    # Degreek számlálása
    degrees_user = {u: {r: 0 for r in relation_to_id} for u in user_to_id}
    degrees_item = {i: {r: 0 for r in relation_to_id} for i in item_to_id}

    for user, rel, item in triplets:
        degrees_user[user][rel] += 1
        degrees_item[item][rel] += 1

    # Feature mátrixok építése
    for u, uid in user_to_id.items():
        user_features[uid] = np.array([degrees_user[u][r] for r in relation_to_id])
    for i, iid in item_to_id.items():
        item_features[iid] = np.array([degrees_item[i][r] for r in relation_to_id])

    return {
        "triplets": triplets,
        "user_to_id": user_to_id,
        "item_to_id": item_to_id,
        "relation_to_id": relation_to_id,
        "user_features": user_features,
        "item_features": item_features
    }


def build_user_item_hetero_data(triplet_dict):
    print("build_user_item_hetero_data started")

    triplets = triplet_dict["triplets"]
    user_to_id = triplet_dict["user_to_id"]
    item_to_id = triplet_dict["item_to_id"]
    relation_to_id = triplet_dict["relation_to_id"]

    user_features = triplet_dict["user_features"]
    item_features = triplet_dict["item_features"]

    data = HeteroData()
    data["user"].x = torch.tensor(user_features, dtype=torch.float)
    data["item"].x = torch.tensor(item_features, dtype=torch.float)

    # --- Kapcsolatok relációnként ---
    edges_by_rel = defaultdict(list)
    for user, rel, item in triplets:
        edges_by_rel[rel].append((user_to_id[user], item_to_id[item]))

    for rel_type, pairs in edges_by_rel.items():
        src, dst = zip(*pairs)
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # előre- és visszairány
        data[("user", rel_type, "item")].edge_index = edge_index
        data[("item", f"rev_{rel_type}", "user")].edge_index = edge_index.flip(0)

    return data

import os
from tqdm import tqdm

def generate_and_save_negatives(hetero_data, rel_to_id, output_path, num_negatives=100):
    """
    Minden user-item élhez (pozitív triplethez) legenerál num_negatives negatív itemet.
    A fájlba mentett struktúra:
        { (head_idx, rel_idx, tail_idx): [neg_item_1, ..., neg_item_N] }
    """
    hetero_data = hetero_data.cpu()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    negatives = []
    head_idx_list, rel_idx_list, tail_idx_list = [], [], []
    for (src_type, rel_type, dst_type), edge_store in hetero_data.edge_index_dict.items():
        if rel_type in rel_to_id.keys():
            edge_index = edge_store.T
            num_edges = edge_index.size(0)
            rel_ids = torch.full((num_edges,), fill_value=rel_to_id[rel_type], dtype=torch.long)

            head_idx_list.append(edge_index[:, 0])
            rel_idx_list.append(rel_ids)
            tail_idx_list.append(edge_index[:, 1])

    head_idx = torch.cat(head_idx_list)
    rel_idx = torch.cat(rel_idx_list)
    tail_idx = torch.cat(tail_idx_list)
    num_items = hetero_data["item"].num_nodes

    print(f"Generating {num_negatives} negatives per positive...")
    for i in tqdm(range(len(head_idx))):
        negatives_for_triplet = torch.randperm(num_items)[:num_negatives]
        # ne tartalmazza a valódi pozitív tail-t
        negatives_for_triplet = negatives_for_triplet[negatives_for_triplet != tail_idx[i]][:num_negatives]
        negatives.append(negatives_for_triplet)

    torch.save({
        "head_idx": head_idx,
        "rel_idx": rel_idx,
        "tail_idx": tail_idx,
        "negatives": negatives
    }, output_path)
    print(f"✅ Saved negative samples to {output_path}")


if __name__ == "__main__":
    path_to_dataset_dir = Path("./inductive_datasets/beibei/")
    path_to_train = path_to_dataset_dir / "ind-train.tsv"
    path_to_validation = path_to_dataset_dir / "ind-dev.tsv"
    path_to_test = path_to_dataset_dir / "ind-test.tsv"

    path_to_datasets = {"train": path_to_train, "validation": path_to_validation, "test": path_to_test}

    train_triplet_dict = load_user_item_triplets(path_to_datasets.get("train"))
    test_triplet_dict = load_user_item_triplets(path_to_datasets.get("test"))