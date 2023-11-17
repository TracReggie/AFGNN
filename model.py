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


class MH_ATT(torch.nn.Module):
    def __init__(self, fea_dim, num_feas, num_heads, E_dropout, att_dropout, adaptive):
        super(MH_ATT, self).__init__()

        self.fea_dim = fea_dim
        self.num_feas = num_feas
        self.num_heads = num_heads

        self.E = MLP(fea_dim * num_feas, fea_dim * num_heads, fea_dim * num_heads, 3, E_dropout, 'prelu')
        self.s = torch.nn.Parameter(torch.FloatTensor(num_heads, 2 * fea_dim, 1))
        self.out_layer = Linear(num_heads * fea_dim, fea_dim)
        self.att_dropout = torch.nn.Dropout(att_dropout)

        self.act = torch.nn.Sigmoid()
        self.adaptive = adaptive

    def reset_parameters(self):
        self.E.reset_parameters()
        self.out_layer.reset_parameters()
        torch.nn.init.normal_(self.s, std=0.01)

    def att_scores(self, batch_fea, mean_fea):  # [b, num_fea, fea_dim]
        batch_att = batch_fea if self.adaptive else mean_fea
        cat_vector = batch_att.reshape(batch_att.size(0), -1)  # [b, num_fea * fea_dim]

        E = self.E(cat_vector)  # [b, fea_dim * num_heads]
        E = E.reshape(-1, self.num_heads, self.fea_dim)  # [b, num_heads, fea_dim]
        E_repeat = E.unsqueeze(2).repeat(1, 1, self.num_feas, 1)  # [b, num_heads, num_fea, fea_dim]

        batch_att = batch_att.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [b, num_heads, num_fea, fea_dim]

        cat_vector = torch.cat([E_repeat, batch_att], dim=3)  # [b, num_heads, num_fea, 2 * fea_dim]

        s = self.s.unsqueeze(0).repeat(batch_att.size(0), 1, 1, 1)  # [b, num_heads, 2 * fea_dim, 1]
        
        att_scores = self.act(torch.matmul(cat_vector, s))  # [b, num_heads, num_fea, 1]
        att_scores = F.softmax(att_scores - 0.5, dim=2)  # [b, num_heads, num_fea, 1]

        return att_scores

    def forward(self, batch_fea, batch_mean):
        att_scores = self.att_scores(batch_fea, batch_mean)  # [b, num_heads, num_fea, 1]
        att_scores = self.att_dropout(att_scores)

        fea_repeat = batch_fea.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  #[b, num_heads, num_fea, fea_dim]

        output = fea_repeat * att_scores
        output = torch.sum(output, dim=2)  # [b, num_heads, fea_dim]
        output = output.reshape(-1, self.fea_dim * self.num_heads)
        output = self.out_layer(output)

        return output


class AFGNN(torch.nn.Module):
    def __init__(
            self,
            fea_dim,
            num_feas,
            num_heads,
            att_dropout,
            hid_dim,
            out_dim,
            pred_layers,
            dropout,
            adaptive
    ):
        super(AFGNN, self).__init__()

        self.mh_att = MH_ATT(fea_dim, num_feas, num_heads, dropout, att_dropout, adaptive)
        self.pred = MLP(fea_dim, hid_dim, out_dim, pred_layers, dropout, 'prelu')

    def reset_parameters(self):
        self.mh_att.reset_parameters()
        self.pred.reset_parameters()

    def attention_scores(self, batch_fea, batch_mean):
        return self.mh_att.att_scores(batch_fea, batch_mean).squeeze()

    def forward(self, batch_fea, batch_mean):
        att_out = self.mh_att(batch_fea, batch_mean)
        out = self.pred(att_out)

        return out
