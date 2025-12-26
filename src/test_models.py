import random
import numpy as np
import os
import wandb
from pathlib import Path
import torch
from tqdm import tqdm
from baseline_models import RandomBaseline, MostPopularBaseline
from multi_node_models import HeteroSAGE_KGE, HGT_KGE
from prepare_data_util_multi_node import load_user_item_triplets, build_user_item_hetero_data
from torch_geometric.loader import NeighborLoader


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

#todo save later
def save_model_scores(model, hetero_data, device, out_path, negatives_path):
    data_dict = torch.load(negatives_path)
    head_idx = data_dict["head_idx"].to(device)
    rel_idx = data_dict["rel_idx"].to(device)
    tail_idx = data_dict["tail_idx"].to(device)
    negatives = data_dict["negatives"]

    hetero_data = hetero_data.to(device)
    model.eval()

    # encode
    with torch.no_grad():
        h_dict = model.encoder(hetero_data.x_dict, hetero_data.edge_index_dict)

    ordered_types = ["user", "item"]
    entity_emb_full = torch.cat([h_dict[t] for t in ordered_types], dim=0)

    offsets = {}
    off = 0
    for nt in ordered_types:
        offsets[nt] = off
        off += h_dict[nt].size(0)

    head_idx_global = head_idx + offsets["user"]
    tail_idx_global = tail_idx + offsets["item"]

    all_scores = []

    for i in tqdm(range(len(head_idx))):
        h = head_idx_global[i]
        r = rel_idx[i]
        t = tail_idx_global[i]
        neg_tails = negatives[i].to(device) + offsets["item"]

        candidates = torch.cat([t.view(1), neg_tails])
        scores = model.kge_head(
            h.expand_as(candidates),
            r.expand_as(candidates),
            candidates,
            node_emb=entity_emb_full
        ).cpu()

        all_scores.append(scores)

    torch.save({"scores": all_scores}, out_path)

def train_evaluate(config=None):
    with wandb.init(config=config):
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(f"./best_models_test_{config.model}", exist_ok=True)
        os.makedirs(f"./scores_{config.model}", exist_ok=True)

        for run_id in range(5,10):
            # --- Set seed ---
            seed = torch.randint(0, 10_000_000, (1,)).item()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            print(f"=== RUN {run_id}/9 | SEED = {seed} ===")

            ##################################################################
            # Modell inicializálás
            ##################################################################
            if config.model == "hgt_transe":
                model = HGT_KGE(
                    metadata=train_hetero_data.metadata(),
                    in_dim=in_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    num_entities=num_entities,
                    num_relations=len(rel_to_id),
                    rel_to_id=rel_to_id,
                    scorer="transe",
                    dropout=config.dropout,
                    num_heads=config.num_heads
                ).to(device)

            elif config.model == "sage_transe":
                model = HeteroSAGE_KGE(
                    metadata=train_hetero_data.metadata(),
                    in_dim=in_dim,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    num_entities=num_entities,
                    num_relations=len(rel_to_id),
                    rel_to_id=rel_to_id,
                    scorer="transe",
                    dropout=config.dropout
                ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

            ##################################################################
            # Dataloaderek
            ##################################################################
            train_loader = NeighborLoader(
                train_hetero_data,
                num_neighbors={key: [10, 10] for key in train_hetero_data.edge_types},
                input_nodes=("user", torch.arange(train_hetero_data["user"].num_nodes)),
                batch_size=config.batch_size,
                shuffle=True
            )

            val_loader = NeighborLoader(
                validation_hetero_data,
                num_neighbors={key: [10, 10] for key in validation_hetero_data.edge_types},
                input_nodes=("user", torch.arange(validation_hetero_data["user"].num_nodes)),
                batch_size=config.batch_size,
                shuffle=False
            )

            ##################################################################
            # TANÍTÁS
            ##################################################################
            best_val = float('inf')
            patience = 2
            patience_counter = 0
            nepochs = 501#501
            for epoch in range(1, nepochs):
                model.train()
                total_loss = 0

                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    loss = model.compute_loss(batch, device)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_train = total_loss / len(train_loader)
                print(f"Run {run_id} | Epoch {epoch} | Train loss: {avg_train:.4f}")

                if epoch % 20 == 0:
                    model.eval()
                    val_total = 0
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_batch = val_batch.to(device)
                            val_total += model.compute_loss(val_batch, device).item()
                    avg_val = val_total / len(val_loader)

                    if avg_val < best_val:
                        best_val = avg_val
                        patience_counter = 0
                        torch.save(model.state_dict(), f"./best_models_test_{config.model}/best_model_{wandb.run.name}_run{run_id}.pt")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
            ##################################################################
            # Betöltjük a legjobb modellt
            ##################################################################
            model.load_state_dict(torch.load(f"./best_models_test_{config.model}/best_model_{wandb.run.name}_run{run_id}.pt"))
            model.eval()

            ##################################################################
            # Teszt értékelés: model
            ##################################################################
            test_neg_path = "./inductive_datasets/beibei/negatives/test_negatives.pt"
            with torch.no_grad():
                mrr, hits = model.evaluate_with_saved_negatives(
                    test_hetero_data,
                    device,
                    negatives_path=test_neg_path,
                    k=10
                )

            ##################################################################
            # Teszt pontszámok mentése
            ##################################################################
            #score_path = f"./scores__{config.model}/{wandb.run.name}_run{run_id}_test_scores.pt"
            #save_model_scores(model, test_hetero_data, device, score_path, negatives_path=test_neg_path)


            wandb.log({
                f"run_{run_id}_seed": seed,
                f"run_{run_id}_test_mrr@10": mrr,
                f"run_{run_id}_test_hits@10": hits,
               #f"run_{run_id}_score_file": score_path,
            })


# ----------------------------
# W&B project names
# ----------------------------

#
configs = {
    "hgt_kge": {
        "model": "hgt_transe",
        "hidden_dim": 64,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.2838379501897075,
        "lr": 0.0469242847169765,
        "batch_size": 2048
        }
    # },
    # "sage_kge": {
    #     "model": "sage_transe",
    #     "hidden_dim": 16,
    #     "num_layers": 4,
    #     "dropout": 0.18762659333403217,
    #     "lr": 0.04058722086756364,
    #     "batch_size": 2048
    # }
}

def evaluate_trained_models(config=None):
    with wandb.init(config=config):
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("runs on ",device)
        test_neg_path = "./inductive_datasets/beibei/negatives/test_negatives.pt"
        print(f"Re-evaluate model '{config.model}'")

        if config.model == "hgt_transe":
            best_model_dir = Path("./best_models_test_hgt_transe")
            model = HGT_KGE(
                metadata=train_hetero_data.metadata(),
                in_dim=in_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_entities=num_entities,
                num_relations=len(rel_to_id),
                rel_to_id=rel_to_id,
                scorer="transe",
                dropout=config.dropout,
                num_heads=config.num_heads
            ).to(device)

        if config.model == "sage_transe":
            best_model_dir = Path("./best_models_test_sage_transe")
            model = HeteroSAGE_KGE(
                metadata=train_hetero_data.metadata(),
                in_dim=in_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_entities=num_entities,
                num_relations=len(rel_to_id),
                rel_to_id=rel_to_id,
                scorer="transe",
                dropout=config.dropout
            ).to(device)

        saved_models = sorted(os.listdir(best_model_dir))
        print(saved_models)
        for nth, best_model_state_dict in enumerate(saved_models):
            print(f"Processing {nth}/{len(os.listdir(best_model_dir))} model_state_dict")
            print(f"Loading {best_model_state_dict}..")
            model.load_state_dict(torch.load(best_model_dir/best_model_state_dict))
            model.eval()
            with torch.no_grad():
                mrr, hits = model.evaluate_with_saved_negatives(
                    test_hetero_data,
                    device,
                    negatives_path=test_neg_path
                )
        log_dict = {
            "seed_index": nth,
            "eval/mrr": mrr,
        }

        # Hits több k-re
        for kk, val in hits.items():
            log_dict[f"eval/hits@{kk}"] = val

        wandb.log(log_dict)
        print(f"Logged to wandb: {log_dict}\n")




project_names = {
    "hgt_kge": "beibei_hgt_kge_fixed_FINAL_rerun"
}
#"baselines": "beibei_baselines"
#"sage_kge": "beibei_sage_kge_fixed_FINAL_rerun"
# ----------------------------
# Run train_evaluate for each model
# -----------------------------*
for model_name, config in configs.items():
    wandb.init(project=project_names[model_name], config=config)
    print(f"=== EVALUATE {model_name} ===")
    evaluate_trained_models(config=config)  # your train_evaluate as in your last working version
    #train_evaluate(config=config)  # your train_evaluate as in your last working version
    wandb.finish()

# ----------------------------
# Run baselines separately on test set
# ----------------------------

#todo should run random 10 times and averaged
#todo shuffled lists are also should be saved or at least scores

#wandb.init(project=project_names["baselines"])
# print("=== EVALUATING BASELINES ===")
#
# test_neg_path = "./inductive_datasets/beibei/negatives/test_negatives.pt"
# test_neg_data = torch.load(test_neg_path)
# test_heads = test_neg_data["head_idx"]
# test_tails = test_neg_data["tail_idx"]
# test_rels = test_neg_data["rel_idx"]
# test_negatives = test_neg_data["negatives"]
#
# rb = RandomBaseline()
# mp = MostPopularBaseline(train_triplet_dict)
#
# rr_rb, hits_rb = [], []
# rr_mp, hits_mp = [], []
#
# for i in range(len(test_heads)):
#     t = test_tails[i].item()
#     candidates = [t] + test_negatives[i].tolist()  # 100-tail list
#
#     mrr, hit = rb.score_and_hits(t, candidates, k=10)
#     rr_rb.append(mrr)
#     hits_rb.append(hit)
#
#     mrr, hit = mp.score_and_hits(t, candidates, k=10)
#     rr_mp.append(mrr)
#     hits_mp.append(hit)
#
# wandb.log({
#     "baseline_random_mrr@10": sum(rr_rb)/len(rr_rb),
#     "baseline_random_hits@10": sum(hits_rb)/len(hits_rb),
#     "baseline_mp_mrr@10": sum(rr_mp)/len(rr_mp),
#     "baseline_mp_hits@10": sum(hits_mp)/len(hits_mp)
# })
# wandb.finish()