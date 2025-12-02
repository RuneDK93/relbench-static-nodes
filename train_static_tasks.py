
import numpy as np
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss
from relbench.datasets import get_dataset
from relbench.tasks import get_task
import os
import math
import numpy as np
from tqdm import tqdm
import torch
from typing import List, Optional, Dict
from torch import Tensor
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.modeling.graph import make_pkey_fkey_graph
from torch_geometric.seed import seed_everything
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from torch_geometric.loader import NeighborLoader
from relbench.modeling.utils import get_stype_proposal
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss

import pickle


# for model
import copy
from typing import Any, Dict, List
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd

from utils.utils import RelBenchModel, get_loaders, GloveTextEmbedding, make_col_stats

from relbench.base import AutoCompleteTask


# Training params
n_epochs = 200
learning_rate = 1e-3 
step_size = 1000                            # n_epochs before reducing lr (high number to not use)
gamma = 0.5                                  # factor to reduce lr by 
batch_size =  128  

# Model params
temporal_strategy = 'last'                 # Sample strategy. Either uniform or last, which will focus on neighbours close in time to seed node
num_neighbours = [128 for i in range(2)]  # The number of neighbours to sample per depth level

num_layers =2                              # Number of layers in GNN model
channels = 128                             # Number of channels in GNN model
aggr = "mean"                              # Aggregator function in GNN model
temporal_encoding = False                  # Whether to use temporal encoding in the model (only for GraphSAGE + neighbor loader currently)

w_decay = 1e-2
# Hgt params
hgt_heads = 16

# Batch Sampler 
#loader_type = 'neighbor' #'neighbor'  # 'neighbor' or 'hgt'. OBS hgt loader is NOT time aware
#model_type = 'graphsage'  # or 'graphsage' / 'hgt

loader_type = 'neighbor' #'neighbor'  # 'neighbor' or 'hgt'. OBS hgt loader is NOT time aware
model_type = 'graphsage'  # or 'graphsage' / 'hgt


# Task param
data_name = 'rel-hm'
task_name = 'customer-age-nth'
#task_name = 'article-index-nth'
#task_name = 'article-colour-nth'


# Advanced inductive temporal method for computing train statistics that also tracks non-time-stamped-nodes related to time-stamped-nodes. Currently only available for rel-hm dataset.
# if False then train stats are computed using all time-stamped nodes present in the training data, and ALL non-time-stamped nodes. 
adv_compute_train_stats = True  

# Print more info. Advised for debugging
verbose = True 


dataset = get_dataset(data_name, download=False) 
task = get_task(data_name, task_name, download=False)

# Using _get_table for uncached
train_table = task._get_table("train") 
val_table = task._get_table("val")
test_table = task._get_table("test")

target_col_table_name = task.entity_table
target_col_name = task.target_col
tasktype = task.task_type.value

# Print what we are working on
print(f'\nWorking on task {task_name} ({tasktype}) for dataset {data_name}...')


# Check wether the task requires removing columns from the input features.
# If so, add the column names to remove_columns
if getattr(task, 'time_independent_node_task', False):
    remove_columns = task.remove_columns
elif isinstance(task, AutoCompleteTask):
    remove_columns = task.remove_columns
    remove_columns.append(task.target_col) # For autocompletetask manually add the target col to remove_columns
else:
    remove_columns = []
if len(remove_columns) > 0:
    print(f'\nNote: This is a special node property task that requires removing columns {remove_columns} from the input data to avoid leakage.')


if tasktype == 'regression':
    loss_fn = L1Loss()
    #loss_fn = MSELoss()    
    tune_metric = "mae"
    higher_is_better = False
    out_channels = 1    

if tasktype == 'binary_classification':
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True

if tasktype == 'multiclass_classification':
    out_channels = task.num_labels
    loss_fn = CrossEntropyLoss()
    tune_metric = "accuracy" 
    higher_is_better = True

# Some book keeping
seed_everything(42)


train_table = task._get_table("train") 
val_table = task._get_table("val")
test_table = task._get_table("test")


if verbose == True:
    print('\n\nTraining table view:')
    print(train_table.df)
    print('\nVal table shape:')
    print(val_table.df.shape)
    print('\nTest table shape:')
    print(test_table.df)
if verbose == True and tasktype == 'multiclass_classification' or tasktype == 'binary_classification':
    print('training target distribution (head 10)')
    print(train_table.df[target_col_name].value_counts().head(10))
    print('val target distribution (head 10)')
    print(val_table.df[target_col_name].value_counts().head(10))
    print('test target distribution (head 10)')
    print(test_table.df[target_col_name].value_counts().head(10))
    print('unique labels train')
    print(train_table.df[target_col_name].unique())
    print('unique labels val')
    print(val_table.df[target_col_name].unique())
    print('unique labels test')
    print(test_table.df[target_col_name].unique())
if verbose == True and tasktype == 'regression':
    print("Train targets: min, max, mean, std:", train_table.df[target_col_name].min(),
    train_table.df[target_col_name].max(), train_table.df[target_col_name].mean(),
    train_table.df[target_col_name].std())
    print("Value counts (top 5):")
    print(train_table.df[target_col_name].value_counts().head(5))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if verbose == True:
    print('device',device)  
root_dir = "./data"


text_embedder_cfg = TextEmbedderConfig(
    text_embedder=GloveTextEmbedding(device=device), batch_size=256
)


# ==================================== Build Graph from Dataset ====================================
print('Building or Loading Full Graph')
db_full = dataset.get_db(upto_test_timestamp=False) # Setting upto_test_time = False to get FULL database
# Make dictionary with column types for full dataset
col_to_stype_dict = get_stype_proposal(db_full) 

# If task is task requires removing columns from input data then remove specified input features from the input data to avoid leakage
# AutoCompleteTask should do this automatically but we double check here that it is actually done.
if len(remove_columns) > 0:
    # Make a dictionary of input features where some columns are removed. 
    # Later use this dictionary when instantiating the graph
    col_to_stype_dict_clean = {
        table_name: {
            col: col_type
            for col, col_type in cols.items()
            if not (table_name == target_col_table_name and col in remove_columns)
        }
        for table_name, cols in col_to_stype_dict.items()
    }

    stype_dict_to_use = col_to_stype_dict_clean # The modified data with features removed
else: # If not a time-independent special task then use full data (note: Autocomplete task will automatically remove the required features from the data)
    stype_dict_to_use = col_to_stype_dict # The original full data



# Instantiate the FULL graph to be used for training/validation/testing. The SAMPLER or LOADER will be responsible for not sampling validation or test data during training.
data_full, full_stats = make_pkey_fkey_graph(
    db_full,
    col_to_stype_dict=stype_dict_to_use,  # speficied column types 
    text_embedder_cfg=text_embedder_cfg,  # our chosen text encoder
    cache_dir=os.path.join(            # Careful about caching! Make sure we are using the correct version of the graph for modelling 
        root_dir, f"{data_name}_{task_name}_full_cache" # store materialized graph for convenience
    ),  
)
print('Building or Loading Train Graph and Train Col Stats')
data_train, train_col_stats_dict = make_pkey_fkey_graph(
    db_full.upto(dataset.val_timestamp - pd.Timedelta("1ns")),  # only use data up to val timestamp for training statistics
    col_to_stype_dict=stype_dict_to_use, 
    text_embedder_cfg=text_embedder_cfg, 
    cache_dir=os.path.join(root_dir, f"{data_name}_{task_name}_train_cache")            
    )


if verbose == True:
    # Optionally manually inspect the entity table with the target value in to verify it includes the input features we expect
    print(f'Manual inspection of input graph (for table [{target_col_table_name}]):')
    print('Remove columns required for this task:', remove_columns)
    print('IF REMOVE COLUMNS IS REQUIRED FOR THIS TASK, THEN DOUBLE CHECK THAT THESE COLUMNS ARE NOT IN THE DATA BELOW!')
    print(data_full[target_col_table_name].tf)  # shows the TensorFrame of the table containing the target value.



# ==================================== Compute Train Col Stats ====================================
if verbose:
    print('Computing or Loading Train Column Stats')

if adv_compute_train_stats and data_name == 'rel-hm': # Special inductive train stats implemented for H&M dataset currently. If using other dataset make sure to change static_tables and key_map
    # If using this overwrite the simple train_col_stats_dict obtained above
    # Get the col_stats_dict for the training data (used for normalisation)
    train_col_stats_dict = make_col_stats(
        db=db_full,
        timestamp=dataset.val_timestamp,  #up_to_timestamp
        static_tables=["customer", "article"],
        key_map={"customer": "customer_id", "article": "article_id"},
        col_to_stype_dict=stype_dict_to_use,
        text_embedder_cfg=text_embedder_cfg,  # our chosen text encoder
        cache_dir=os.path.join(root_dir, f"{data_name}_{task_name}_train_cache")
    )

# ================================================================================================

# ==================================== Get Data Loaders and model ====================================


loader_dict_train = get_loaders(
    data=data_train,
    task=task,
    tables= {"train": train_table},
    num_neighbors=num_neighbours,
    batch_size=batch_size,
    temporal_strategy=temporal_strategy,
    loader_type=loader_type,  
    num_workers=0,
)

tables_inference = {
    "val": val_table,
    "test": test_table,
}
loader_dict_inference = get_loaders(
    data=data_full,
    task=task,
    tables=tables_inference,
    num_neighbors=num_neighbours,
    batch_size=batch_size,
    temporal_strategy=temporal_strategy,
    loader_type=loader_type,  
    num_workers=0,
)

model = RelBenchModel(
    model_type=model_type,
    loader_type=loader_type,    
    data=data_train,
    col_stats_dict=train_col_stats_dict,
    num_layers=num_layers,
    channels=channels,
    out_channels=out_channels,
    aggr=aggr,
    norm="batch_norm",
    hgt_heads=hgt_heads,
    temporal_encoding=temporal_encoding,
).to(device)
# ================================================================================================




# ==================================== Train and Test functions ====================================
# Train and test 
def train(loader) -> float:
    model.train()

    # Values for computing train accuracy
    if tasktype != 'regression':
        total = 0
        correct = 0

    loss_accum = count_accum = 0
    for batch in tqdm(loader):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        if tasktype=='multiclass_classification' or tasktype=='binary_classification' :
            target = batch[task.entity_table].y.long()


        if tasktype=='regression':
            target = batch[task.entity_table].y.float()   
        

        pred = pred.float()        
        loss = loss_fn(pred, target)   
        
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0) 
        count_accum += pred.size(0)

        if tasktype != 'regression': # compute accuracy for non-regression task
            correct += ( torch.argmax(pred, dim=1)==target ).sum().item()
            total += len(target)

    if tasktype=='regression':
        return loss_accum / count_accum
    else:
        train_acc = correct / total        
        return loss_accum / count_accum , train_acc



@torch.no_grad()
def test(loader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()

# Training params
#optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate) 
optimizer =  torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay) 
epochs = n_epochs 

# Lr Scheduler
# Reduce LR by factor gamma every step_size epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# ================================================================================================

# ==================================== TRAINING ==================================================
print(f'Will train for {n_epochs} epochs with lr {learning_rate} and batch_size {batch_size}')
print(f'Model type: {model_type}')
print(f'sampler type: {loader_type}')
if model_type == 'hgt':
    print(f'HGT heads: {hgt_heads}')
print('Num neighbours:',num_neighbours) 
print('Channels:',channels)   
print('Layers:',num_layers)   
print('Weigh Decay:',w_decay)
print('Starting to train...')
state_dict = None
best_val_metric = -math.inf if higher_is_better else math.inf
train_loss_list = []
val_metrics_list = []
test_metrics_list = []


for epoch in range(1, epochs + 1):
    if tasktype == 'regression':
        train_loss = train(loader_dict_train['train'])
    else:
        _,train_loss = train(loader_dict_train['train']) # defining train loss value as accuracy for non-regression task (for printing)
     
    scheduler.step()      # step shceduler

    val_pred = test(loader_dict_inference["val"])
    val_metrics = task.evaluate(val_pred, val_table)

    test_pred = test(loader_dict_inference["test"])
    test_metrics = task.evaluate(test_pred,test_table)  # Manually set test table.

    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}, Test metrics: {test_metrics}")

    # Save metrics per epoch
    train_loss_list.append(train_loss)
    val_metrics_list.append(val_metrics)
    test_metrics_list.append(test_metrics)

    if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
        not higher_is_better and val_metrics[tune_metric] < best_val_metric
    ):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())

model.load_state_dict(state_dict)
val_pred = test(loader_dict_inference["val"])
val_metrics = task.evaluate(val_pred, val_table)
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict_inference["test"])
#test_metrics = task.evaluate(test_pred)
test_metrics = task.evaluate(test_pred,test_table) # Manually set test table. 
print(f"Best test metrics: {test_metrics}")
# ================================================================================================




# ========================================  Save results   =======================================
# Save results
# Create the output folder
output_dir = f"result/{task_name}_{n_epochs}_{learning_rate}"
os.makedirs(output_dir, exist_ok=True)

# test targets
targets_path = os.path.join(output_dir, "test_targets.csv")
test_table.df[target_col_name].to_csv(targets_path, index=False)

# test predictions
preds_path = os.path.join(output_dir, "test_predictions.csv")
# If test_pred is a numpy array, convert to DataFrame

if tasktype=='regression':
    preds_df = pd.DataFrame(test_pred, columns=["predictions"])
    preds_df.to_csv(preds_path, index=False)


if tasktype=='multiclass_classification' or tasktype=='binary_classification' :
# Convert test predictions to numpy
    if isinstance(test_pred, torch.Tensor):
        test_pred = test_pred.cpu().numpy()
    pred_class = np.argmax(test_pred, axis=1)
    # Save predictions
    preds_df = pd.DataFrame(pred_class, columns=["predictions"])
    preds_df.to_csv(preds_path, index=False)


# save training curves
# Save training loss
pd.DataFrame({"train_loss": train_loss_list}).to_csv(
    os.path.join(output_dir, "train_loss.csv"), index_label="epoch"
)

# val metrics
pd.DataFrame(val_metrics_list).to_csv(
    os.path.join(output_dir, "val_metrics.csv"), index_label="epoch"
)

# test metrics
pd.DataFrame(test_metrics_list).to_csv(
    os.path.join(output_dir, "test_metrics.csv"), index_label="epoch"
)

print(f"Saved test targets to {targets_path}")
print(f"Saved test predictions to {preds_path}")
# ================================================================================================
