from ast import literal_eval
from argparse import ArgumentParser
import copy
import json
from tqdm import tqdm
from pathlib import Path
import pickle as pkl
from numpy import percentile
import torch
from models.gnn.graph_classifier import GraphClassifier, GraphClassifierv2
from typing import Optional
from torch import device, tensor, save as torch_save, Tensor, where
from torch.cuda import is_available as cuda_is_available
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_isolated_nodes



def parse_arguments():
    parser = ArgumentParser(description='Train a graph classifier')

    # Add the required arguments
    parser.add_argument('--train_file_path', type=str, required=True, help='Path to the training file')
    parser.add_argument('--test_file_path', type=str, required=True, help='Path to the test file')
    parser.add_argument('--model_output_dir', type=str, required=True, help='Directory to save the trained model and the edge threshold')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs to train')

    # Add the optional arguments
    parser.add_argument('--hidden_channel_dimensions', type=str, default="[128, 128]", help='List of integers for hidden channel dimensions as a literal string such as "[int_0, int_1]]" (default: [128, 128])')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size to use during training (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate to use during training (default: 5e-5)')
    parser.add_argument('--edge_weight_percentile', type=float, default=80, help='Percentage or sequence of percentages for the percentiles to compute. Values must be between 0 and 100 inclusive (default=0)')
    parser.add_argument('--dropout', type=float, default=0.75, help='Dropout default 0.75')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer default SGD')
    parser.add_argument('--early_stopping_patience', type=int, default=75, help='early_stopping_patience default 5')
    

    return parser.parse_args()


def load_pytorch_geometric_data(file_path:str) -> list:
    data_format_string = "The data is in an unexpected format. The function assumes the file path is a pickled list of torch_geometric.data.Data objects"
    with open(file_path, 'rb') as f:
        dataset = pkl.load(f)
    # assert isinstance(dataset, list), data_format_string
    # for data in dataset:
    #     assert isinstance(data, Data), data_format_string
    dataset = [data for data in dataset if data]
    return dataset         


def edge_filter(pyg_dataset, edge_weight_percentile:float, threshold:Optional[float]=None) -> float:
    if not threshold:
        edge_weights = []
        for data in pyg_dataset:
            edge_weights += data.edge_attr.flatten().tolist()
        threshold = percentile(edge_weights, edge_weight_percentile)
    for data in pyg_dataset:
        edge_attr = tensor(data.edge_attr)
        valid_edges = where(edge_attr > tensor(threshold))[0]
        data.edge_index = data.edge_index[:,valid_edges]
        # data.edge_index = data.edge_index[:,valid_edges][[1,0],:]
        data.edge_attr = edge_attr[valid_edges]
        
        # remove isolated nodes
        data.edge_index, data.edge_attr, mask = remove_isolated_nodes(data.edge_index, data.edge_attr, num_nodes=data.x.shape[0])
        data.x = data.x[mask,:]
        data.node_types = data.node_types[mask]
        
    return threshold

def train_graph_classifier(
    train_file_path:str,
    test_file_path:str,
    model_output_dir:str,
    num_epochs:int,
    hidden_channel_dimensions:list=[128,128], # note GNNs can incur over-smoothing issues when the number of layers is too high. Consider the diameter of your graph when picking these values.
    batch_size:int=4,
    learning_rate:float=5e-5,
    edge_weight_percentile:int=0,
    dropout:float=0.75,
    optimizer_type:str='SGD',
    early_stopping_patience:int=10
    
):
    # load_data assuming data is a list of pytorch geometric DATA
    train_dataset = load_pytorch_geometric_data(train_file_path)
    test_dataset = load_pytorch_geometric_data(test_file_path)

    no_improvement_counter=0
    best_test_acc=0.0
    # filter edges beneath edge_weight_percentile and remove isolated nodes
# ask leo
    # import pdb
    # pdb.set_trace()
    
    for graph in train_dataset:
        assert(graph.edge_index.shape[1]==graph.edge_attr.shape[0])
    for graph in test_dataset:
        assert(graph.edge_index.shape[1]==graph.edge_attr.shape[0])        
    if "edge_attr" in train_dataset[0] and edge_weight_percentile != 0:
        edge_weight_threshold = edge_filter(train_dataset, edge_weight_percentile, threshold=None)
        edge_weight_threshold = edge_filter(test_dataset, edge_weight_percentile, threshold=edge_weight_threshold)
    else:
        edge_weight_threshold = 0
    
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # determine whether a gpu is available
    torch_device = device("cuda") if cuda_is_available() else device("cpu")
    
    # instantiate model, evaluation loss, and optimizer
    model_metadata = {
        "hidden_channel_dimensions": [train_dataset[0].x.shape[1]] + hidden_channel_dimensions, # dynamically creates gatconv layers and pulls dataset feature size
        "num_classes": len(set([data.y for data in train_dataset])) # find distinct labels in training dataset
    }
    model = GraphClassifier(**model_metadata).to(torch_device)
    criterion = CrossEntropyLoss().to(torch_device)
    if optimizer_type=='SGD':
        optimizer = SGD(model.parameters(), lr=learning_rate, nesterov=False, momentum=0.9)
    elif optimizer_type=='adam':
        optimizer = Adam(model.parameters(), lr=learning_rate)

    def train():
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            cloned_data = copy.deepcopy(data).to(torch_device) # Clone data and delete to avoid GPU memory issues)
            cloned_data.x = cloned_data.x.to(torch.float32)
            out = model(
                node_features=cloned_data.x,
                edge_index=cloned_data.edge_index,
                batch=cloned_data.batch,
                dropout_percentage=0.75,
                # node_subset_indices=where(cloned_data.node_types==tensor([5]).to(torch_device))[0]  # Perform a single forward pass.
            )
            loss = criterion(out, cloned_data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            # optimizer = SGD(model.parameters(), lr=learning_rate, nesterov=False, momentum=0.9)
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            del cloned_data

    def test(loader):
        model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            cloned_data = copy.deepcopy(data).to(torch_device) # Clone data and delete to avoid GPU memory issues
            cloned_data.x = cloned_data.x.to(torch.float32)
            out = model(
                node_features=cloned_data.x,
                edge_index=cloned_data.edge_index,
                batch=cloned_data.batch,
                # node_subset_indices=where(cloned_data.node_types==tensor([5]).to(torch_device))[0]
            ) # Perform a single forward pass.
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == cloned_data.y).sum())  # Check against ground-truth labels.
            del cloned_data
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.
        
        
    for epoch in tqdm(range(0, num_epochs)):
        train()
        if epoch % 20 == 0:
            torch.cuda.empty_cache()
            train_acc = test(train_loader)
            torch.cuda.empty_cache()
            test_acc = test(test_loader)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}. no_improvement_counter={no_improvement_counter}')
            if test_acc>best_test_acc:
                best_test_acc=test_acc
                no_improvement_counter=0
            else:
                no_improvement_counter+=1

            if no_improvement_counter>=early_stopping_patience:
                print(f"Early stopping. Best Test Accuracy:{best_test_acc}")
                break
    # run on final training epoch
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # save model
    model.to(device("cpu")) # move model back to cpu prior to saving
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    torch_save(model.state_dict(), Path(model_output_dir, "model.pt"))  
    model_metadata['edge_weight_threshold'] = edge_weight_threshold
    with open(Path(model_output_dir, "model_metadata.json"), 'w') as f:
        json.dump(model_metadata, f)

def main():
    args = parse_arguments()
    train_graph_classifier(
        train_file_path=args.train_file_path,
        test_file_path=args.test_file_path,
        model_output_dir=args.model_output_dir,
        num_epochs=args.num_epochs,
        hidden_channel_dimensions=literal_eval(args.hidden_channel_dimensions),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        edge_weight_percentile=args.edge_weight_percentile,
        dropout=args.dropout,
        optimizer_type=args.optimizer,
        early_stopping_patience=args.early_stopping_patience

    )

if __name__ == "__main__":
    main()

# example usage
# python -m model_training.graph_classification --train_file_path=/home/ec2-user/lbetthauser/data/orchestration/hybrid_orchestration/spl/v1/gnn_inputs/train_data.pkl --test_file_path=/home/ec2-user/lbetthauser/data/orchestration/hybrid_orchestration/spl/v1/gnn_inputs/test_data.pkl --model_output_dir=/home/ec2-user/lbetthauser/models/orchestrator/spl/v1/test_model --num_epochs=100 --hidden_channel_dimensions=[512,32] --batch_size=16 --edge_weight_percentile=95
# python -m model_training.graph_classification --train_file_path=/home/ec2-user/lbetthauser/data/orchestration/hybrid_orchestration/security/v0/gnn_inputs/train_data.pkl --test_file_path=/home/ec2-user/lbetthauser/data/orchestration/hybrid_orchestration/security/v0/gnn_inputs/test_data.pkl --model_output_dir=/home/ec2-user/lbetthauser/models/orchestrator/spl/v0/test_model_2 --num_epochs=100 --hidden_channel_dimensions=[512,32] --batch_size=16 --edge_weight_percentile=95
# python -m model_training.graph_classification --train_file_path=/home/ec2-user/lbetthauser/data/orchestration/hybrid_orchestration/spl/v2/gnn_inputs/train_data.pkl --test_file_path=/home/ec2-user/lbetthauser/data/orchestration/hybrid_orchestration/spl/v2/gnn_inputs/test_data.pkl --model_output_dir=/home/ec2-user/lbetthauser/models/orchestrator/spl/v2/test_model_2 --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=16 --edge_weight_percentile=90 --learning_rate=1e-4



# python -m model_training.graph_classification --train_file_path=/home/ec2-user/prachi/saia-finetuning/grading/data/v1/train_data.pkl --test_file_path=/home/ec2-user/prachi/saia-finetuning/grading/data/v1/test_data.pkl --model_output_dir=/home/ec2-user/prachi/leo/saia-finetuning/models/trained_models/v1/ --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=16 --edge_weight_percentile=90 --learning_rate=1e-4


# python -m model_training.graph_classification --train_file_path=/home/ec2-user/prachi/saia-finetuning/grading/data/v2/train_data.pkl --test_file_path=/home/ec2-user/prachi/saia-finetuning/grading/data/v2/test_data.pkl --model_output_dir=/home/ec2-user/prachi/leo/saia-finetuning/models/trained_models/v2/ --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=16 --edge_weight_percentile=95 --learning_rate=1e-4

# python3 -m model_training.graph_classification --train_file_path=grading/data/v2/train_data.pkl --test_file_path=grading/data/v2/test_data.pkl --model_output_dir=grading/data/models --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=16 --edge_weight_percentile=95 --learning_rate=1e-4

# python3 -m model_training.graph_encoding_contrastive --train_file_path=grading/data/v2/train_data.pkl --test_file_path=grading/data/v2/test_data.pkl --model_output_dir=grading/data/models --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=64 --edge_weight_percentile=0 --learning_rate=1e-3
# python3 -m model_training.graph_classification --train_file_path=grading/data/llama/train_data.pkl --test_file_path=grading/data/llama/test_data.pkl --model_output_dir=grading/data/models --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=64 --edge_weight_percentile=0 --learning_rate=1e-3

# python3 -m model_training.graph_classification --train_file_path=grading/data/llama/paper/train_data.pkl --test_file_path=grading/data/llama/paper/test_data.pkl --model_output_dir=grading/data/models --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=32 --edge_weight_percentile=0 --learning_rate=1e-4

# python3 -m model_training.graph_classification --train_file_path=grading/data/llama/paper/train_data_h.pkl --test_file_path=grading/data/llama/paper/test_data_h.pkl --model_output_dir=grading/data/models --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=32 --edge_weight_percentile=0 --learning_rate=1e-4

# python3 -m model_training.graph_classification --train_file_path=grading/data/llama/paper/train_data_tok1.pkl --test_file_path=grading/data/llama/paper/test_data_tok1.pkl --model_output_dir=grading/data/models/tok_1_llama --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=32 --edge_weight_percentile=0 --learning_rate=1e-4

# python3 -m model_training.graph_classification --train_file_path=grading/data/llama/paper/train_data_jailbrk1.pkl --test_file_path=grading/data/llama/paper/test_data_jailbrk1.pkl --model_output_dir=grading/data/models/ --num_epochs=1000 --hidden_channel_dimensions=[64] --batch_size=32 --edge_weight_percentile=0 --learning_rate=1e-4