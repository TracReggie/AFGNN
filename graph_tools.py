import torch
from torch_sparse import SparseTensor


def GCN_adj(edge_index, self_loops=True):
    adj = SparseTensor(row=edge_index[0], col=edge_index[1])
    adj = adj.set_diag() if self_loops else adj

    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    adj_sym = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    adj_rw = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj

    adj_sym = adj_sym.to_scipy(layout='csr')
    adj_rw = adj_rw.to_scipy(layout='csr')

    return adj_sym, adj_rw


def get_fea_list(data, kl, km, kh, pl=1.5, al=1, pm=-1.2, am=1, ph=1.5, ah=0.5, res=True):
    adj_sym, _ = GCN_adj(data.edge_index)

    fea_list = [data.x] if res else []

    xl, xm, xh = data.x, data.x, data.x

    fea_list_l, fea_list_m, fea_list_h = [], [], []

    for _ in range(kl):
        xl = pl * (al * torch.from_numpy(adj_sym @ xl) + (1 - al) * xl)
        fea_list_l.append(xl)

    for _ in range(km):
        xm = pm * (torch.from_numpy(adj_sym @ (adj_sym @ xm)) - am * xm)
        fea_list_m.append(xm)

    for _ in range(kh):
        xh = ph * ((-ah) * torch.from_numpy(adj_sym @ xh) + (1 - ah) * xh)
        fea_list_h.append(xh)

    fea_list = fea_list + fea_list_l + fea_list_m + fea_list_h

    return fea_list
