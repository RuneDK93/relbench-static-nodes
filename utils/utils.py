import torch
from torch import nn, Tensor
from torch_geometric.nn import MLP, HGTConv
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch_geometric.data import HeteroData
from torch_frame.data.stats import StatType
from typing import Any, Dict, List, Optional
from torch.nn import ModuleDict, Embedding


# ===== Model =====
class RelBenchModel(nn.Module):
    def __init__(
        self,
        model_type: str,                         # 'hgt' or 'graphsage'
        loader_type: str,                        # 'hgt' or 'neighbor'
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[str, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str = "sum",
        norm: str = "batch_norm",
        hgt_heads: int = 8,
        temporal_encoding: bool = False,
        use_node_id_embeddings: bool = False,     # transductive option
        node_types_with_embeddings=None,          # which node types to embed
    ):
        super().__init__()

        self.model_type = model_type.lower()
        assert self.model_type in ["hgt", "graphsage"], f"Unknown model_type {model_type}"

        self.loader_type = loader_type.lower()
        assert self.loader_type in ["hgt", "neighbor"], f"Unknown loader_type {loader_type}"

        self.use_node_id_embeddings = use_node_id_embeddings

        self.temporal_encoding = temporal_encoding

        # Shared encoders
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]],
            channels=channels,
        )

        # optional shallow node ID embeddings (for transductive mode) (i.e. node index embeddings)
        if self.use_node_id_embeddings:
            if node_types_with_embeddings is None:
                node_types_with_embeddings = data.node_types  # default: all node types
            self.embedding_dict = ModuleDict({
                node_type: Embedding(data[node_type].num_nodes, channels)
                for node_type in node_types_with_embeddings
            })
        else:
            self.embedding_dict = None

        # Select GNN type
        if self.model_type == "graphsage":
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr=aggr,
                num_layers=num_layers,
            )
        else:  # HGT
            self.gnn = nn.ModuleList([
                HGTConv(
                    in_channels=channels,
                    out_channels=channels,
                    metadata=(data.node_types, data.edge_types),
                    heads=hgt_heads,
                ) for _ in range(num_layers)
            ])

            # optional per-type layernorm after each HGT layer
            self.layernorm_dict = nn.ModuleDict({
                nt: nn.LayerNorm(channels) for nt in data.node_types
            })


        # Output head
        self.head = MLP(
            in_channels=channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()

        if isinstance(self.gnn, nn.ModuleList):
            for layer in self.gnn:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        else:
            self.gnn.reset_parameters()

        self.head.reset_parameters()

        if self.embedding_dict is not None:
            for emb in self.embedding_dict.values():
                nn.init.normal_(emb.weight, std=0.1)

    def forward(
        self,
        batch: HeteroData,
        entity_table: str,
    ) -> Tensor:
        """
        Forward pass works for batches sampled with NeighborLoader or HGTLoader.
        """

        x_dict = self.encoder(batch.tf_dict)

        # Determine number of seed nodes in the batch
        if self.loader_type == "neighbor":        
            num_seed_nodes = batch.num_sampled_nodes_dict[entity_table][0] # the first entry here is the number of seed nodes at depth 0 (before sampling)
        else:  # HGT
            num_seed_nodes = batch[entity_table]['batch_size']


        # Temporal encoding
        if self.temporal_encoding and self.loader_type == "neighbor":
            seed_time = batch[entity_table].seed_time
            rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] += rel_time

        # Node ID embeddings
        if self.embedding_dict is not None:
            for node_type, emb in self.embedding_dict.items():
                if "n_id" in batch[node_type]:
                    x_dict[node_type] += emb(batch[node_type].n_id)

        # Message passing
        if self.model_type == "graphsage" and self.loader_type == "neighbor":
            x_dict = self.gnn(
                x_dict,
                batch.edge_index_dict,
                getattr(batch, "num_sampled_nodes_dict", None),
                getattr(batch, "num_sampled_edges_dict", None),
            )

        elif self.model_type == "graphsage" and self.loader_type == "hgt":
            x_dict = self.gnn(
                x_dict,
                batch.edge_index_dict,
            )            

        #else:  # simpler HGT without skip connections (residuals) and layernorm
#            for conv in self.gnn:
#                x_dict = conv(x_dict, batch.edge_index_dict)

        else:  # HGT
            for conv in self.gnn:
                x_res = x_dict  # save for residual connection
                x_dict = conv(x_dict, batch.edge_index_dict)

                # residual add (per type)
                x_dict = {
                    k: x_dict[k] + x_res[k]
                    for k in x_dict.keys()
                }

                #per-type LayerNorm
                if hasattr(self, "layernorm_dict"):
                    x_dict = {
                        k: self.layernorm_dict[k](x_dict[k])
                        for k in x_dict.keys()
                    }        

        # return predictions for seed nodes only
        return self.head(x_dict[entity_table][: num_seed_nodes])    


# ===== Loaders =====
from torch_geometric.loader import NeighborLoader, HGTLoader
from typing import Union
from relbench.modeling.graph import get_node_train_table_input

def get_loaders(
    data: HeteroData,
    task,
    tables: Dict[str, "Table"],
    num_neighbors: Union[List[int], Dict[str, List[int]]],
    batch_size: int = 256,
    temporal_strategy: str = "last",
    loader_type: str = "neighbor",  # "neighbor" or "hgt"
    num_workers: int = 0,
) -> Dict[str, Union[NeighborLoader, HGTLoader]]:
    """
    Build dataloaders for train/val/test with configurable loader type.
    """
    loader_dict = {}

    for split, table in tables.items():
        table_input = get_node_train_table_input(table=table, task=task)

        if loader_type.lower() == "neighbor":
            loader = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                input_nodes=table_input.nodes,
                input_time=table_input.time,
                transform=table_input.transform,
                batch_size=batch_size,
                temporal_strategy=temporal_strategy,
                shuffle=False,#(split == "train"),
                num_workers=num_workers,
                persistent_workers=False,
                time_attr="time",
            )
        elif loader_type.lower() == "hgt":
            loader = HGTLoader(
                data,
                num_samples=num_neighbors,
                input_nodes=table_input.nodes,
                batch_size=batch_size,
                is_sorted=False,
                transform=table_input.transform,                
                num_workers=num_workers,
            )
        else:
            raise ValueError(f"Unknown loader_type {loader_type}")

        loader_dict[split] = loader

    return loader_dict



# ==================================== Text Encodeer  ====================================
from sentence_transformers import SentenceTransformer

class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )
    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))
    

# ================================================================================================
    



# ==================================== Temporally Aware Train stats function  ====================================
# Function to compute feature statistics for a graph up to a specific timepoint. This ALSO respects static tables without explicit timestamps (like customers, articles in H&M data)
# Only nodes who had interactions UP to specified timestamp are included. Important to avoid information leakage when computing col_stats for normalisation
import os
import pandas as pd
import numpy as np
import pickle
from torch_frame.data import Dataset
from relbench.base import Database, Table
from relbench.modeling.utils import remove_pkey_fkey
from torch_frame import stype


def make_col_stats(
    db: "Database",
    timestamp: pd.Timestamp,
    static_tables,
    key_map,
    col_to_stype_dict,
    text_embedder_cfg,
    cache_dir: Optional[str] = None,
) -> Dict:
    """
    Compute column stats up to a given timestamp, optionally caching the result.
    """
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"train_col_stats_adv.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    # Filter timestamped tables as usual
    new_table_dict = {
        name: table.upto(timestamp) for name, table in db.table_dict.items()
    }

    # Identify active entities in timestamped tables
    active_entities = {}
    for table_name, table in new_table_dict.items():
        if table.time_col is not None:
            for st in static_tables or []:
                key_col = key_map.get(st)
                if key_col and key_col in table.df.columns:
                    active_entities.setdefault(st, set()).update(table.df[key_col].unique())

    # Filter static tables based on active entities
    for st in static_tables or []:
        if st in new_table_dict and st in active_entities:
            table = new_table_dict[st]
            df_filtered = table.df[table.df[key_map[st]].isin(active_entities[st])]
            df_filtered = df_filtered.reset_index(drop=True)
            df_filtered[table.pkey_col] = np.arange(len(df_filtered))
            new_table_dict[st] = Table(
                df=df_filtered,
                pkey_col=table.pkey_col,
                fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                time_col=None
            )

    # Compute col_stats using dataset
    col_stats_dict = {}
    for table_name, table in Database(table_dict=new_table_dict).table_dict.items():
        df = table.df
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()
        col_to_stype = col_to_stype_dict[table_name]
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:
            col_to_stype = {"__const__": stype.numerical}
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=None)

        col_stats_dict[table_name] = dataset.col_stats

    # Cache the result for convenience
    if cache_dir is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(col_stats_dict, f)

    return col_stats_dict
# ================================================================================================
