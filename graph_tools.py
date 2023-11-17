import torch
from torch_sparse import SparseTensor


def GCN_adj(edge_index, self_loops=True):
    adj = SparseTensor(row=edge_index[0], col=edge_index[1])
    adj = adj.set_diag() if self_loops else adj

    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    adj_sym = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    adj_sym = adj_sym.to_scipy(layout='csr')

    return adj_sym


def get_features(data, kl, km, kh, pl=1.5, al=1, pm=-1.2, am=1, ph=1.5, ah=0.5, res=True):
    adj_sym = GCN_adj(data.edge_index)
    features = [data.x] if res else []
    xl, xm, xh = data.x, data.x, data.x

    for _ in range(kl):
        xl = pl * (al * torch.from_numpy(adj_sym @ xl) + (1 - al) * xl)
        features.append(xl)

    for _ in range(km):
        xm = pm * (torch.from_numpy(adj_sym @ (adj_sym @ xm)) - am * xm)
        features.append(xm)

    for _ in range(kh):
        xh = ph * ((-ah) * torch.from_numpy(adj_sym @ xh) + (1 - ah) * xh)
        features.append(xh)

    features = torch.cat(features, dim=1)
    features = features.view(data.num_nodes, 1+kl+km+kh, -1)

    return features
