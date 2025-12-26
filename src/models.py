import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.nn.kge import TransE, DistMult
from tqdm import tqdm
from torch_geometric.nn import SAGEConv, HeteroConv

# --- HGT encoder ---
class HGTEncoder(nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels, num_layers=2, num_heads=2, dropout=0.2):
        super().__init__()
        node_types, edge_types = metadata
        self.lin_dict = nn.ModuleDict({
            ntype: nn.Linear(in_channels, hidden_channels) for ntype in node_types
        })
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata=metadata, heads=num_heads)
            for _ in range(num_layers)
        ])
        self.act = nn.GELU()
        self.norms = nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types})
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        h = {ntype: self.lin_dict[ntype](x) for ntype, x in x_dict.items()}
        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            for ntype, h_nt in h_new.items():
                h_nt = self.act(h_nt)
                h_nt = self.norms[ntype](h_nt)
                h_nt = F.dropout(h_nt, self.dropout, training=self.training)
                h[ntype] = h_nt
        return h



class TransEBatch(nn.Module):
    def __init__(self,num_nodes, num_relations, embedding_dim):
        super().__init__()
        self.num_rel = num_relations
        self.embedding_dim = embedding_dim
        self.rel_emb = nn.Parameter(torch.randn(num_relations, embedding_dim))
        self.num_nodes = num_nodes

    def forward(self, head_idx, rel_idx, tail_idx, node_emb):
        # node_emb: teljes entitás embedding mátrix [num_entities, embedding_dim]
        head_e = node_emb[head_idx]  # [batch, dim]
        tail_e = node_emb[tail_idx]  # [batch, dim]
        rel_e = self.rel_emb[rel_idx]  # [batch, dim]
        score = -torch.norm(head_e + rel_e - tail_e, p=2, dim=-1)  # negatív L2 norm
        return score

    @torch.no_grad()
    def random_sample(self, head_idx, rel_idx, tail_idx, replace_tail_only=True):
        num_samples = head_idx.size(0)

        head_neg = head_idx.clone()
        tail_neg = tail_idx.clone()

        if replace_tail_only:
            # a tail-ek batchen belüli permutációja, hogy cserélje a tail-ek pozícióját a batchen
            perm = torch.randperm(num_samples, device=tail_idx.device)
            tail_neg = tail_idx[perm]
        else:
            num_neg = num_samples // 2
            rnd = torch.randint(0, self.num_nodes, (num_samples,), device=head_idx.device)#fixme if not only tail change is needed
            head_neg[:num_neg] = rnd[:num_neg]
            tail_neg[num_neg:] = rnd[num_neg:]

        #print("tail_neg max=", tail_neg.max())
        return head_neg, rel_idx, tail_neg


    @torch.no_grad()
    def evaluate_with_saved_negatives(self, hetero_data, device, negatives_path, k=10):
        """
        Evaluate model using pre-generated negative tails from disk.
        """
        data_dict = torch.load(negatives_path)
        head_idx = data_dict["head_idx"].to(device)
        rel_idx = data_dict["rel_idx"].to(device)
        tail_idx = data_dict["tail_idx"].to(device)
        negatives = data_dict["negatives"]

        hetero_data = hetero_data.to(device)
        h_dict = self.encoder(hetero_data.x_dict, hetero_data.edge_index_dict)
        entity_emb_full = torch.cat([h for h in h_dict.values()], dim=0)

        hits, reciprocal_ranks = [], []
        print(f"Evaluating with {len(head_idx)} triples and saved negatives...")

        for i in tqdm(range(len(head_idx))):
            h = head_idx[i]
            r = rel_idx[i]
            t = tail_idx[i]
            neg_tails = negatives[i].to(device)

            # Candidate = pozitív tail + negatív tailok
            candidates = torch.cat([t.view(1), neg_tails])
            scores = self.kge_head(h.expand_as(candidates), r.expand_as(candidates), candidates, node_emb=entity_emb_full)
            rank = int((scores.argsort(descending=True) == 0).nonzero().view(-1))  # pozíció a listában (0 a pozitív)
            reciprocal_ranks.append(1.0 / (rank + 1))
            hits.append(rank < k)

        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = float(torch.tensor(hits, dtype=torch.float).mean())

        print(f"✅ MRR: {mrr:.4f} | Hits@{k}: {hits_at_k:.4f}")
        return mrr, hits_at_k

# --- HGT + KGE head combined ---

# --- Heterogeneous GraphSAGE encoder ---
class HeteroSAGEEncoder(nn.Module):
    def __init__(self, metadata, in_channels, hidden_channels, num_layers=2, dropout=0.2):
        super().__init__()
        node_types, edge_types = metadata

        # Lineáris előkódolás minden node típushoz
        self.lin_dict = nn.ModuleDict({
            ntype: nn.Linear(in_channels, hidden_channels)
            for ntype in node_types
        })

        # HeteroConv rétegek (tuple kulcsokkal!)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            convs = {
                (src, rel, dst): SAGEConv((-1, -1), hidden_channels)
                for src, rel, dst in edge_types
            }
            self.convs.append(HeteroConv(convs, aggr="mean"))

        self.norms = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_channels)
            for ntype in node_types
        })
        self.dropout = dropout
        self.act = nn.GELU()

    def forward(self, x_dict, edge_index_dict):
        # Bemeneti lineáris transzformáció
        # print("forward x_dict=",x_dict)
        # print("forward edge_index_dict=",edge_index_dict)
        h = {ntype: self.lin_dict[ntype](x) for ntype, x in x_dict.items()}

        # Rétegenként végigmegyünk a HeteroConv blokkokon
        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            for ntype, h_nt in h_new.items():
                h_nt = self.act(h_nt)
                h_nt = self.norms[ntype](h_nt)
                h_nt = F.dropout(h_nt, self.dropout, training=self.training)
                h[ntype] = h_nt
        return h

