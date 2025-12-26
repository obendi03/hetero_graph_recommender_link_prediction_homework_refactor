import os
import subprocess

import torch
import wandb
from multi_node_models import HGT_KGE
from pathlib import Path
import torch
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

train_hetero_data = train_hetero_data.to(device)
validation_hetero_data = validation_hetero_data.to(device)

num_entities = num_users + num_items  # teljes “globális” entitástér
rel_to_id = train_triplet_dict["relation_to_id"]


def train_evaluate(config=None):
    with wandb.init(config=config):
        config = wandb.config
        print(config)
        # --- Device ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("runs on ",device)
        # --- Modell ---
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

        elif  config.model == "sage_transe":
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
        epochs = 500 #todo change to 500
        best_val_loss = float('inf')
        patience = 2
        patience_counter = 0

        # --- DataLoader ---
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
        train_losses = []
        val_losses = []
        # --- Training loop ---
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = model.compute_loss(batch, device)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch:02d} | Avg Loss: {avg_train_loss:.8f}")
            # --- Validation ---
            if epoch % 20 == 0:
                model.eval()
                val_total_loss = 0.0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = val_batch.to(device)
                        val_loss = model.compute_loss(val_batch, device)
                        val_total_loss += val_loss.item()
                avg_val_loss = val_total_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                print(f"Epoch {epoch:02d} | Avg VAL Loss: {avg_val_loss:.8f}")
            # --- Early stopping & save best model ---
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f"./best_models/best_model_{wandb.run.name}.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        model.eval()
        with torch.no_grad():
            mrr, hits = model.evaluate_with_saved_negatives(
                validation_hetero_data,
                device,
                negatives_path="./inductive_datasets/beibei/negatives/val_negatives.pt",
                k=10
            )
        torch.save(model.state_dict(), f"./best_models/best_model_{wandb.run.name}.pt")
        # --- Log final metrics ---
        wandb.log({
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "val_mrr@10": mrr,
            "val_hits@10": hits
        })

sweep_config = {
    "kgt_kge": {
        "method": "bayes",  # Bayesian optimization
        "metric": {"name": "val_mrr@10", "goal": "maximize"},
        "parameters": {
            "model": {"value": "hgt_transe"},
            "lr": {"min": 1e-3, "max": 1e-1},
            "hidden_dim": {"values": [16, 32, 64]},
            "num_layers": {"values": [2, 3, 4, 5]},
            "dropout": {"min": 0.1, "max": 0.3},
            "batch_size": {"values": [2048]},
            "num_heads": {"values": [2, 4]}
        }
    },
    "sage_kge": {
        "method": "bayes",  # Bayesian optimization
        "metric": {"name": "val_mrr@10", "goal": "maximize"},
        "parameters": {
            "model": {"value": "sage_transe"},
            "lr": {"min": 1e-3, "max": 1e-1},
            "hidden_dim": {"values": [16, 32, 64]},
            "num_layers": {"values": [2, 3, 4, 5]},
            "dropout": {"min": 0.1, "max": 0.3},
            "batch_size": {"values": [2048]},
        }
    }
}
#
for model_name, config in sweep_config.items():
    sweep_id = wandb.sweep(config, project=f"{model_name}_sweep_hetero_graph_v9_2")
    wandb.agent(sweep_id, function=train_evaluate, count=10)
# sweep_ids = list(["obendi2003-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/kgt_kge_sweep_hetero_graph_v3/h307sp98",
#                   "obendi2003-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/sage_kge_sweep_hetero_graph_v3/vwwz5osx"])
# i = 0
# for model_name, config in sweep_config.items():
#     # sweep_id = wandb.sweep(config, project=f"{model_name}_sweep_hetero_graph_v3")
#     wandb.agent(sweep_ids[i], function=train_evaluate, count=10)
#     i += 1
#
