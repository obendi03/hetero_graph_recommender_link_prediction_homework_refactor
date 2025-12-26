from pathlib import Path
from tqdm import tqdm
import torch
import wandb
import random
import pickle
from collections import Counter

from baseline_models import MostPopularBaseline, RandomBaseline
from prepare_data_util_multi_node import load_user_item_triplets, build_user_item_hetero_data

path_to_dataset_dir = Path("./inductive_datasets/beibei/")
path_to_train = path_to_dataset_dir / "ind-train.tsv"
path_to_validation = path_to_dataset_dir / "ind-dev.tsv"
path_to_test =  path_to_dataset_dir / "ind-test.tsv"

path_to_datasets = {"train": path_to_train,"validation": path_to_validation,"test":path_to_test}


train_triplet_dict = load_user_item_triplets(path_to_datasets.get("train"))
validation_triplet_dict = load_user_item_triplets(path_to_datasets.get("validation"))
test_triplet_dict = load_user_item_triplets(path_to_datasets.get("test"))

train_hetero_data = build_user_item_hetero_data(train_triplet_dict)
validation_hetero_data = build_user_item_hetero_data(validation_triplet_dict)
test_hetero_data = build_user_item_hetero_data(test_triplet_dict)


train_metadata = train_hetero_data.metadata()

num_users = len(train_triplet_dict["user_to_id"])
num_items = len(train_triplet_dict["item_to_id"])
num_relations = len(train_triplet_dict["relation_to_id"])

# a két típus különböző méretű lehet, de a dimenzió egyezik
in_dim_user = train_hetero_data["user"].x.size(1)
in_dim_item = train_hetero_data["item"].x.size(1)

assert in_dim_user == in_dim_item, "User and item feat dims should match"

in_dim = in_dim_user

train_hetero_data["user"].nid = torch.arange(train_hetero_data["user"].num_nodes)
train_hetero_data["item"].nid = torch.arange(train_hetero_data["item"].num_nodes)

validation_hetero_data["user"].nid = torch.arange(validation_hetero_data["user"].num_nodes)
validation_hetero_data["item"].nid = torch.arange(validation_hetero_data["item"].num_nodes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("runs on",device)

train_hetero_data = train_hetero_data.to(device)
validation_hetero_data = validation_hetero_data.to(device)

num_entities = num_users + num_items  # teljes “globális” entitástér
rel_to_id = train_triplet_dict["relation_to_id"]
# -----------------------
# EVALUÁCIÓ 10 seed-del
# -----------------------
wandb.init(project="beibei_baselines_FINAL_v4")
print("=== EVALUATING BASELINES ===")

test_neg_path = "./inductive_datasets/beibei/negatives/test_negatives.pt"
test_neg_data = torch.load(test_neg_path)
test_heads = test_neg_data["head_idx"]
test_tails = test_neg_data["tail_idx"]
test_rels = test_neg_data["rel_idx"]
test_negatives = test_neg_data["negatives"]

# 10 seed
import pickle
import torch

# ======== kapcsoló ========
use_inductive_mask = False   # vagy False

#todo megcsinalni hogy ne csak taileket mszkoljon

#todo most common itemnél meg kell nézni hogy mely index alapjan kell a legnepszerubbet megfeleltetni
#todo a tripletsnél az eredeti formában vannak annyi a dolgom hogy
#1. kikeresem a legnepszerubb id-t a trainben, ez a tesztben mas lesz es a tesztbeli megfeleltetését használom


# ======== ha kell: maszkolás előkészítése ========
if use_inductive_mask:
    train_items = set(train_triplet_dict["item_to_id"].values())

    inductive_mask = torch.tensor(
        [(t.item() not in train_items) for t in test_tails],
        dtype=torch.bool
    )

    test_heads_masked = test_heads[inductive_mask]
    test_tails_masked = test_tails[inductive_mask]
    test_rels_masked  = test_rels[inductive_mask]
    test_negatives_masked = test_negatives[inductive_mask]

else:
    test_heads_masked = test_heads
    test_tails_masked = test_tails
    test_rels_masked  = test_rels
    test_negatives_masked = test_negatives

# ======== baseline futtatás ========

seeds = [42, 123, 2025, 7, 999, 555, 2023, 31415, 2718, 8888]

all_rb_results = []
all_rb_shuffles = []

mp = MostPopularBaseline()
k_list = [1, 3, 5, 10]

wandb.init(project="baseline_runs_FINAL_hopefully_v2", name="combined_baselines")

# ===================== RANDOM BASELINE (ALL SEEDS IN ONE RUN) =====================

for run, seed in enumerate(seeds):

    rb = RandomBaseline(seed=seed)

    rr_rb_full, rr_rb_10 = [], []
    hits_rb = {k: [] for k in k_list}

    print(f"running random baseline seed {seed} ({run+1}/{len(seeds)})")

    for i in tqdm(range(len(test_heads_masked))):
        t = test_tails_masked[i].item()
        candidates = [t] + test_negatives_masked[i].tolist()

        mrr_full, mrr_at_10, hits, _ = rb.score_and_hits(t, candidates, k_list)

        rr_rb_full.append(mrr_full)
        rr_rb_10.append(mrr_at_10)

        for k in k_list:
            hits_rb[k].append(hits[k])

    # AVERAGE VALUES
    seed_mrr_full = sum(rr_rb_full) / len(rr_rb_full)
    seed_mrr_at_10 = sum(rr_rb_10) / len(rr_rb_10)
    seed_hits_avg = {k: sum(hits_rb[k]) / len(hits_rb[k]) for k in k_list}

    # LOG EVERYTHING FOR THIS SEED IN ONE RUN
    wandb.log({
        f"random/mrr_full/seed_{seed}": seed_mrr_full,
        f"random/mrr@10/seed_{seed}": seed_mrr_at_10,
    })

    for k in k_list:
        wandb.log({
            f"random/hits@{k}/seed_{seed}": seed_hits_avg[k]
        })


# ========================= MOST POPULAR BASELINE ============================

rr_mp_full, rr_mp_10 = [], []
hits_mp = {k: [] for k in k_list}

print("Running MostPopularBaseline...")

for i in tqdm(range(len(test_heads_masked))):
    t = test_tails_masked[i].item()
    candidates = [t] + test_negatives_masked[i].tolist()

    mrr_full, mrr_at_10, hits = mp.score_and_hits(t, candidates, k_list)

    rr_mp_full.append(mrr_full)
    rr_mp_10.append(mrr_at_10)

    for k in k_list:
        hits_mp[k].append(hits[k])

# LOG MP METRICS (still same run)
wandb.log({
    "mp/mrr_full": sum(rr_mp_full) / len(rr_mp_full),
    "mp/mrr@10": sum(rr_mp_10) / len(rr_mp_10),
})

for k in k_list:
    wandb.log({
        f"mp/hits@{k}": sum(hits_mp[k]) / len(hits_mp[k])
    })

wandb.finish()
