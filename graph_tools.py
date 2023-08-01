import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import get_laplacian, add_self_loops, to_scipy_sparse_matrix


def GCN_adj(edge_index, self_loops=True):
    adj = SparseTensor(row=edge_index[0], col=edge_index[1])
    adj = adj.set_diag() if self_loops else adj

    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    adj_sym = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    adj_rw = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj

    return adj_sym, adj_rw


def get_fea_list(data, low_order, mid_order, high_order, pl=1.5, al=1, pm=-1.2, am=1, ph=1.5, ah=0.5, res=True):
    adj_sym, _ = GCN_adj(data.edge_index)
    adj_sym = adj_sym.to_scipy(layout='csr')

    fea_list = [data.x] if res else []

    xl, xm, xh = data.x, data.x, data.x

    fea_list_l, fea_list_m, fea_list_h = [], [], []

    for _ in range(low_order):
        xl = pl * (al * torch.from_numpy(adj_sym @ xl) + (1 - al) * xl)
        fea_list_l.append(xl)

    for _ in range(mid_order):
        xm = pm * (torch.from_numpy(adj_sym @ (adj_sym @ xm)) - am * xm)
        fea_list_m.append(xm)

    for _ in range(high_order):
        xh = ph * ((-ah) * torch.from_numpy(adj_sym @ xh) + (1 - ah) * xh)
        fea_list_h.append(xh)

    fea_list = fea_list + fea_list_l + fea_list_m + fea_list_h

    return fea_list


def get_fea_list2(data, low_order, mid_order, high_order, res=True, self_loops=True,
                  pl=1.5, al=1, pm=-1.2, am=1, ph=1.5, ah=0.5):
    if self_loops:
        adj, _ = add_self_loops(data.edge_index)
    else:
        adj = data.edge_index

    l_sym = get_laplacian(adj, normalization='sym')
    scipy_l = to_scipy_sparse_matrix(l_sym[0], l_sym[1])

    # print(l_sym, scipy_l)

    fea_list = [data.x] if res else []

    xl, xm, xh = data.x, data.x, data.x

    fea_list_l, fea_list_m, fea_list_h = [], [], []

    for _ in range(low_order):
        xl = pl * (-al * torch.from_numpy(scipy_l @ xl) + xl)
        fea_list_l.append(xl)

    for _ in range(mid_order):
        xm = pm * (torch.from_numpy(scipy_l @ (scipy_l @ xm)) - 2 * torch.from_numpy(scipy_l @ xm) + (1 - am) * xm)
        fea_list_m.append(xm)

    for _ in range(high_order):
        xh = ph * (ah * torch.from_numpy(scipy_l @ xh) + (1 - 2 * ah) * xh)
        fea_list_h.append(xh)

    fea_list = fea_list + fea_list_l + fea_list_m + fea_list_h

    return fea_list


def get_cheb_list(data, hops):
    adj, _ = add_self_loops(data.edge_index)
    l_sym = get_laplacian(adj, normalization='sym')
    scipy_l = to_scipy_sparse_matrix(l_sym[0], l_sym[1])

    x0 = data.x
    x1 = torch.from_numpy(scipy_l @ x0) - x0

    fea_list = [x0, x1]
    if hops < 3:
        pass
    else:
        for _ in range(hops - 2):
            xi = 2 * torch.from_numpy(scipy_l @ fea_list[-1]) - 3 * fea_list[-2]
            fea_list.append(xi)

    return fea_list
