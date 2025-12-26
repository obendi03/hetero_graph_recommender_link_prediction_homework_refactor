import torch

from models import HGTEncoder, TransEBatch, HeteroSAGEEncoder
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class HeteroSAGE_KGE(nn.Module):
    def __init__(self, metadata, in_dim, hidden_dim, num_layers,
                 num_entities, num_relations, rel_to_id, scorer="transe", dropout=0.2):
        super().__init__()

        self.encoder = HeteroSAGEEncoder(metadata, in_dim, hidden_dim, num_layers, dropout)
        self.num_relations = num_relations
        self.rel_to_id = rel_to_id
        self.hidden_dim = hidden_dim

        if scorer.lower() == "transe":
            self.kge_head = TransEBatch(num_entities, num_relations, hidden_dim)
        else:
            raise ValueError("Supported scorers: 'transe' or 'distmult'")

        self.kge_head.node_emn = nn.Parameter(torch.zeros(num_entities, hidden_dim), requires_grad=True)
        self.num_entities = num_entities

    def forward(self, hetero_data):
        h_dict = self.encoder(hetero_data.x_dict, hetero_data.edge_index_dict)
        all_embs = torch.cat([h for h in h_dict.values()], dim=0)
        self.kge_head.node_emn.data.copy_(all_embs)
        return h_dict

    def compute_loss(self, hetero_data, device):
        hetero_data = hetero_data.to(device)

        # 1️⃣ Encode node-ok
        h_dict = self.encoder(hetero_data.x_dict, hetero_data.edge_index_dict)

        # 2️⃣ Explicit sorrend és concat
        ordered_types = ["user", "item"]
        entity_emb_full = torch.cat([h_dict[nt] for nt in ordered_types], dim=0)

        # 3️⃣ Offset számítás a lokális -> globális indexhez
        offsets = {}
        offset = 0
        for nt in ordered_types:
            offsets[nt] = offset
            offset += h_dict[nt].size(0)

        # 4️⃣ Tripletek globális indexekkel
        head_idx_list, rel_idx_list, tail_idx_list = [], [], []
        for (src_type, rel_type, dst_type), edge_store in hetero_data.edge_index_dict.items():
            if rel_type in self.rel_to_id.keys():
                edge_index = edge_store
                head_idx_list.append(edge_index[0] + offsets[src_type])
                tail_idx_list.append(edge_index[1] + offsets[dst_type])
                rel_idx_list.append(torch.full((edge_index.size(1),),
                                               fill_value=self.rel_to_id[rel_type],
                                               device=device))

        pos_head_idx = torch.cat(head_idx_list)
        pos_rel_idx = torch.cat(rel_idx_list)
        pos_tail_idx = torch.cat(tail_idx_list)

        # 5️⃣ Negatív minták
        neg_head_idx, neg_rel_idx, neg_tail_idx = self.kge_head.random_sample(
            pos_head_idx, pos_rel_idx, pos_tail_idx
        )

        # 6️⃣ KGE loss számítása
        pos_scores = self.kge_head(pos_head_idx, pos_rel_idx, pos_tail_idx, node_emb=entity_emb_full)
        neg_scores = self.kge_head(neg_head_idx, neg_rel_idx, neg_tail_idx, node_emb=entity_emb_full)

        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        return loss

    def evaluate_with_saved_negatives(self, hetero_data, device, negatives_path,  k=(1,3,5,10)):
        """
        Evaluate model using pre-generated negative tails with offset-safe embeddings.
        """
        data_dict = torch.load(negatives_path)
        head_idx = data_dict["head_idx"].to(device)
        rel_idx = data_dict["rel_idx"].to(device)
        tail_idx = data_dict["tail_idx"].to(device)
        negatives = data_dict["negatives"]

        hetero_data = hetero_data.to(device)

        if isinstance(k, int):
            k_list = [k]
        else:
            k_list = list(k)

        # 1️⃣ Encode node-ok
        h_dict = self.encoder(hetero_data.x_dict, hetero_data.edge_index_dict)

        # 2️⃣ Explicit sorrend és concat
        ordered_types = ["user", "item"]
        entity_emb_full = torch.cat([h_dict[nt] for nt in ordered_types], dim=0)

        # 3️⃣ Offset számítás
        offsets = {}
        offset = 0
        for nt in ordered_types:
            offsets[nt] = offset
            offset += h_dict[nt].size(0)

        # 4️⃣ Head/Tail globális indexekhez hozzáadjuk az offsetet
        # (feltételezve, hogy head_idx és tail_idx a megfelelő node-típushoz tartozik)
        # pl. ha head user, tail item, akkor head + offsets['user'], tail + offsets['item']
        # Ha a negatív minták is ugyanazon a típuson belül vannak, ugyanígy kezeljük
        # Itt a korábbi mentett negatívok már lokális ID-k, így offsetet kell hozzáadni:
        head_idx_global = head_idx + offsets["user"]
        tail_idx_global = tail_idx + offsets["item"]

        reciprocal_ranks = []
        hits_dict = {kk: [] for kk in k_list}

        print(f"Evaluating with {len(head_idx)} triples and saved negatives...")

        for i in tqdm(range(len(head_idx))):
            h = head_idx_global[i]
            r = rel_idx[i]
            t = tail_idx_global[i]
            neg_tails = negatives[i].to(device) + offsets["item"]  # negatív tail-ek item offsettel

            # Candidate = pozitív tail + negatív tailok
            candidates = torch.cat([t.view(1), neg_tails])
            scores = self.kge_head(
                h.expand_as(candidates),
                r.expand_as(candidates),
                candidates,
                node_emb=entity_emb_full
            )
            rank = int((scores.argsort(descending=True) == 0).nonzero().view(-1))  # pozitív helye
            reciprocal_ranks.append(1.0 / (rank + 1))
            for kk in k_list:
                hits_dict[kk].append(rank < kk)

        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_out = {
            kk: float(torch.tensor(hits_dict[kk], dtype=torch.float).mean())
            for kk in k_list
        }

        print(f"✅ MRR: {mrr:.4f}")
        for kk in k_list:
            print(f"Hits@{kk}: {hits_out[kk]:.4f}")

        return mrr, hits_out

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class HGT_KGE(nn.Module):
    def __init__(self, metadata, in_dim, hidden_dim, num_layers,
                 num_entities, num_relations, rel_to_id, scorer="transe", dropout=0.2, num_heads=2):
        super().__init__()

        # --- Encoder: HGT ---
        self.encoder = HGTEncoder(
            metadata=metadata,
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        self.num_relations = num_relations
        self.rel_to_id = rel_to_id
        self.hidden_dim = hidden_dim

        # --- KGE fej (TransE) ---
        if scorer.lower() == "transe":
            self.kge_head = TransEBatch(num_entities, num_relations, hidden_dim)
        else:
            raise ValueError("Supported scorers: 'transe' only for now")

        self.kge_head.node_emn = nn.Parameter(torch.zeros(num_entities, hidden_dim), requires_grad=True)
        self.num_entities = num_entities

    # ---------------------------------------------------------
    def forward(self, hetero_data):
        h_dict = self.encoder(hetero_data.x_dict, hetero_data.edge_index_dict)
        all_embs = torch.cat([h for h in h_dict.values()], dim=0)
        self.kge_head.node_emn.data.copy_(all_embs)
        return h_dict

    def compute_loss(self, hetero_data, device):
        hetero_data = hetero_data.to(device)

        # 1️⃣ Encode node-ok
        h_dict = self.encoder(hetero_data.x_dict, hetero_data.edge_index_dict)

        # 2️⃣ Explicit sorrend és concat
        ordered_types = ["user", "item"]
        entity_emb_full = torch.cat([h_dict[nt] for nt in ordered_types], dim=0)

        # 3️⃣ Offset számítás a lokális -> globális indexhez
        offsets = {}
        offset = 0
        for nt in ordered_types:
            offsets[nt] = offset
            offset += h_dict[nt].size(0)

        # 4️⃣ Tripletek globális indexekkel
        head_idx_list, rel_idx_list, tail_idx_list = [], [], []
        for (src_type, rel_type, dst_type), edge_store in hetero_data.edge_index_dict.items():
            if rel_type in self.rel_to_id.keys():
                edge_index = edge_store
                head_idx_list.append(edge_index[0] + offsets[src_type])
                tail_idx_list.append(edge_index[1] + offsets[dst_type])
                rel_idx_list.append(torch.full((edge_index.size(1),),
                                               fill_value=self.rel_to_id[rel_type],
                                               device=device))

        pos_head_idx = torch.cat(head_idx_list)
        pos_rel_idx = torch.cat(rel_idx_list)
        pos_tail_idx = torch.cat(tail_idx_list)

        # 5️⃣ Negatív minták
        neg_head_idx, neg_rel_idx, neg_tail_idx = self.kge_head.random_sample(
            pos_head_idx, pos_rel_idx, pos_tail_idx
        )

        # 6️⃣ KGE loss számítása
        pos_scores = self.kge_head(pos_head_idx, pos_rel_idx, pos_tail_idx, node_emb=entity_emb_full)
        neg_scores = self.kge_head(neg_head_idx, neg_rel_idx, neg_tail_idx, node_emb=entity_emb_full)

        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        return loss

    def evaluate_with_saved_negatives(self, hetero_data, device, negatives_path, k=(1,3,5,10)):
        """
        Evaluate model using pre-generated negative tails with offset-safe embeddings.
        """
        data_dict = torch.load(negatives_path)
        head_idx = data_dict["head_idx"].to(device)
        rel_idx = data_dict["rel_idx"].to(device)
        tail_idx = data_dict["tail_idx"].to(device)
        negatives = data_dict["negatives"]

        hetero_data = hetero_data.to(device)

        if isinstance(k, int):
            k_list = [k]
        else:
            k_list = list(k)

        # 1️⃣ Encode node-ok
        h_dict = self.encoder(hetero_data.x_dict, hetero_data.edge_index_dict)

        # 2️⃣ Explicit sorrend és concat
        ordered_types = ["user", "item"]
        entity_emb_full = torch.cat([h_dict[nt] for nt in ordered_types], dim=0)

        # 3️⃣ Offset számítás
        offsets = {}
        offset = 0
        for nt in ordered_types:
            offsets[nt] = offset
            offset += h_dict[nt].size(0)

        # 4️⃣ Head/Tail globális indexekhez hozzáadjuk az offsetet
        # (feltételezve, hogy head_idx és tail_idx a megfelelő node-típushoz tartozik)
        # pl. ha head user, tail item, akkor head + offsets['user'], tail + offsets['item']
        # Ha a negatív minták is ugyanazon a típuson belül vannak, ugyanígy kezeljük
        # Itt a korábbi mentett negatívok már lokális ID-k, így offsetet kell hozzáadni:
        head_idx_global = head_idx + offsets["user"]
        tail_idx_global = tail_idx + offsets["item"]

        reciprocal_ranks = []
        hits_dict = {kk: [] for kk in k_list}
        print(f"Evaluating with {len(head_idx)} triples and saved negatives...")

        for i in tqdm(range(len(head_idx))):
            h = head_idx_global[i]
            r = rel_idx[i]
            t = tail_idx_global[i]
            neg_tails = negatives[i].to(device) + offsets["item"]  # negatív tail-ek item offsettel

            # Candidate = pozitív tail + negatív tailok
            candidates = torch.cat([t.view(1), neg_tails])
            scores = self.kge_head(
                h.expand_as(candidates),
                r.expand_as(candidates),
                candidates,
                node_emb=entity_emb_full
            )
            rank = int((scores.argsort(descending=True) == 0).nonzero().view(-1))  # pozitív helye
            reciprocal_ranks.append(1.0 / (rank + 1))
            for kk in k_list:
                hits_dict[kk].append(rank < kk)

        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_out = {
            kk: float(torch.tensor(hits_dict[kk], dtype=torch.float).mean())
            for kk in k_list
        }

        print(f"✅ MRR: {mrr:.4f}")
        for kk in k_list:
            print(f" Hits@{kk}: {hits_out[kk]:.4f}")

        return mrr, hits_out


    # ---------------------------------------------------------
    # def compute_loss(self, hetero_data, device):
    #     hetero_data = hetero_data.to(device)
    #
    #     # Node embeddingek a batch-re
    #     h_dict = self.encoder(hetero_data.x_dict, hetero_data.edge_index_dict)
    #     entity_emb_full = torch.cat([h for h in h_dict.values()], dim=0)
    #
    #     # --- Tripletek összegyűjtése lokális ID-kkel ---
    #     head_idx_list, rel_idx_list, tail_idx_list = [], [], []
    #     for (src_type, rel_type, dst_type), edge_store in hetero_data.edge_index_dict.items():
    #         if rel_type in self.rel_to_id.keys():
    #             edge_index = edge_store
    #             head_idx_list.append(edge_index[0])
    #             tail_idx_list.append(edge_index[1])
    #             rel_idx_list.append(torch.full((edge_index.size(1),),
    #                                            fill_value=self.rel_to_id[rel_type],
    #                                            device=device))
    #
    #     pos_head_idx = torch.cat(head_idx_list)
    #     pos_rel_idx = torch.cat(rel_idx_list)
    #     pos_tail_idx = torch.cat(tail_idx_list)
    #
    #     # Negatív minták
    #     neg_head_idx, neg_rel_idx, neg_tail_idx = self.kge_head.random_sample(
    #         pos_head_idx, pos_rel_idx, pos_tail_idx
    #     )
    #
    #     # --- KGE loss ---
    #     pos_scores = self.kge_head(pos_head_idx, pos_rel_idx, pos_tail_idx, node_emb=entity_emb_full)
    #     neg_scores = self.kge_head(neg_head_idx, neg_rel_idx, neg_tail_idx, node_emb=entity_emb_full)
    #
    #     labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    #     scores = torch.cat([pos_scores, neg_scores])
    #
    #     loss = F.binary_cross_entropy_with_logits(scores, labels)
    #     return loss

    # ---------------------------------------------------------
    # @torch.no_grad()
    # def evaluate_with_saved_negatives(self, hetero_data, device, negatives_path, k=10):
    #     """
    #     Evaluate model using pre-generated negative tails from disk.
    #     """
    #     data_dict = torch.load(negatives_path)
    #     head_idx = data_dict["head_idx"].to(device)
    #     rel_idx = data_dict["rel_idx"].to(device)
    #     tail_idx = data_dict["tail_idx"].to(device)
    #     negatives = data_dict["negatives"]
    #
    #     hetero_data = hetero_data.to(device)
    #     h_dict = self.encoder(hetero_data.x_dict, hetero_data.edge_index_dict)
    #     entity_emb_full = torch.cat([h for h in h_dict.values()], dim=0)
    #
    #     hits, reciprocal_ranks = [], []
    #     print(f"Evaluating with {len(head_idx)} triples and saved negatives...")
    #
    #     for i in tqdm(range(len(head_idx))):
    #         h = head_idx[i]
    #         r = rel_idx[i]
    #         t = tail_idx[i]
    #         neg_tails = negatives[i].to(device)
    #
    #         # Candidate = pozitív tail + negatív tailok
    #         candidates = torch.cat([t.view(1), neg_tails])
    #         scores = self.kge_head(
    #             h.expand_as(candidates),
    #             r.expand_as(candidates),
    #             candidates,
    #             node_emb=entity_emb_full
    #         )
    #         rank = int((scores.argsort(descending=True) == 0).nonzero().view(-1))  # pozitív helye
    #         if rank < k:
    #             reciprocal_ranks.append(1.0 / (rank + 1))
    #         else:
    #             reciprocal_ranks.append(0.0)
    #         hits.append(rank < k)
    #
    #     mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
    #     hits_at_k = float(torch.tensor(hits, dtype=torch.float).mean())
    #
    #     print(f"✅ MRR: {mrr:.4f} | Hits@{k}: {hits_at_k:.4f}")
    #     return mrr, hits_at_k
