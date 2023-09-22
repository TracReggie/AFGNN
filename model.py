import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout, act='relu'):
        super(MLP, self).__init__()

        self.model = torch.nn.ModuleList()
        self.model.append(Linear(in_dim, hid_dim))

        for _ in range(num_layers - 2):
            self.model.append(Linear(hid_dim, hid_dim))

        self.model.append(Linear(hid_dim, out_dim))

        self.dropout = torch.nn.Dropout(dropout)

        if act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'prelu':
            self.act = torch.nn.PReLU()
        elif act == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def reset_parameters(self):
        for layer in self.model:
            layer.reset_parameters()

    def forward(self, x):
        for layer in self.model[:-1]:
            x = self.dropout(self.act(layer(x)))

        x = self.model[-1](x)

        return x


class MH_ListAtt(torch.nn.Module):
    def __init__(self, fea_dim, num_fea, num_heads, dropout, att_dropout, adaptive=True):
        super(MH_ListAtt, self).__init__()

        self.fea_dim = fea_dim
        self.num_heads = num_heads

        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)

        self.E = MLP(fea_dim * num_fea, fea_dim * num_heads, fea_dim * num_heads, 3, dropout, 'prelu')
        self.s = torch.nn.Parameter(torch.FloatTensor(num_heads, 2 * fea_dim, 1))
        self.out_layer = torch.nn.Linear(fea_dim * num_heads, fea_dim)

        self.act = torch.nn.Sigmoid()
        self.adaptive = adaptive

    def reset_parameters(self):
        self.E.reset_parameters()
        self.out_layer.reset_parameters()
        torch.nn.init.normal_(self.s, std=0.001)

    def att_scores(self, fea_list, mean_list):
        att_list = fea_list if self.adaptive else mean_list
        cat_vector = torch.cat(att_list, dim=1)

        E = self.E(cat_vector)
        E = E.view(-1, self.fea_dim, self.num_heads)  # [num_row, fea_dim, num_heads]
        E = torch.permute(E, [2, 0, 1]).contiguous()  # [num_heads, num_row, fea_dim]

        att_list = [fea.unsqueeze(0).repeat(self.num_heads, 1, 1).contiguous() for fea in att_list]
        # [num_heads, num_row, fea_dim]

        att_scores = [self.act(torch.matmul(torch.cat((E, fea), dim=2), self.s)).view(self.num_heads, -1, 1)
                      for fea in att_list]

        att_scores = torch.cat(att_scores, dim=2)
        att_scores = F.softmax(att_scores - 0.5, dim=2)  # [num_heads, num_row, num_fea]

        return att_scores

    def forward(self, fea_list, mean_list):
        att_scores = self.att_scores(fea_list, mean_list)

        fea_list = [self.dropout(fea) for fea in fea_list]
        fea_list = [fea.unsqueeze(0).repeat(self.num_heads, 1, 1).contiguous() for fea in fea_list]

        att_out = self.att_dropout(att_scores[:, :, 0].view(self.num_heads, -1, 1)) * fea_list[0]

        for i in range(1, len(fea_list)):
            att_out += self.att_dropout(att_scores[:, :, i].view(self.num_heads, -1, 1)) * fea_list[i]
        # att_out: [num_heads, num_row, fea_dim]

        att_out = torch.permute(att_out, [1, 0, 2])  # [num_row, num_heads, fea_dim]
        att_out = torch.reshape(att_out, [-1, self.fea_dim * self.num_heads])
        att_out = self.out_layer(att_out)

        return att_out


class AFGNN(torch.nn.Module):
    def __init__(self, fea_dim, num_fea, num_heads, att_dropout, hid_dim, out_dim, num_layers, dropout, adaptive=True):
        super(AFGNN, self).__init__()

        self.att = MH_ListAtt(fea_dim, num_fea, num_heads, dropout, att_dropout, adaptive)
        self.pred = MLP(fea_dim, hid_dim, out_dim, num_layers, dropout)

    def reset_parameters(self):
        self.att.reset_parameters()
        self.pred.reset_parameters()

    def attention_scores(self, fea_list, mean_list):
        return self.att.att_scores(fea_list, mean_list)

    def forward(self, fea_list, mean_list):
        att_out = self.att(fea_list, mean_list)
        out = self.pred(att_out)

        return out
