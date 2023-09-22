from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import RandomNodeSplit, Compose, ToUndirected
from torch_geometric.utils import index_to_mask


def return_dataset(dataset_name: str):
    if dataset_name == 'cora':
        T = Compose([RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2)])
        dataset = Planetoid(root='/home/zq2/data', name='Cora', transform=T)
        data = dataset[0]

    if dataset_name == 'citeseer':
        T = RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2)
        dataset = Planetoid(root='/home/zq2/data', name='CiteSeer', transform=T)
        data = dataset[0]

    if dataset_name == 'pubmed':
        T = RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2)
        dataset = Planetoid(root='/home/zq2/data', name='PubMed', transform=T)
        data = dataset[0]

    if dataset_name == 'photo':
        T = RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2)
        dataset = Amazon(root='/home/zq2/data', name='Photo', transform=T)
        data = dataset[0]

    if dataset_name == 'computer':
        T = RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2)
        dataset = Amazon(root='/home/zq2/data', name='Computers', transform=T)
        data = dataset[0]

    if dataset_name == 'cs':
        T = RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2)
        dataset = Coauthor(root='/home/zq2/data', name='CS', transform=T)
        data = dataset[0]

    if dataset_name == 'physics':
        T = RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2)
        dataset = Coauthor(root='/home/zq2/data', name='Physics', transform=T)
        data = dataset[0]

    if dataset_name == 'arxiv':
        dataset = PygNodePropPredDataset('ogbn-arxiv', '/home/zq2/data', ToUndirected())
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        data.y = data.y.squeeze(1)
        data.train_mask = index_to_mask(split_idx['train'], data.num_nodes)
        data.val_mask = index_to_mask(split_idx['valid'], data.num_nodes)
        data.test_mask = index_to_mask(split_idx['test'], data.num_nodes)

    if dataset_name == 'products':
        dataset = PygNodePropPredDataset('ogbn-products', '/home/zq2/data')
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        data.y = data.y.squeeze(1)
        data.train_mask = index_to_mask(split_idx['train'], data.num_nodes)
        data.val_mask = index_to_mask(split_idx['valid'], data.num_nodes)
        data.test_mask = index_to_mask(split_idx['test'], data.num_nodes)

    return data, dataset
