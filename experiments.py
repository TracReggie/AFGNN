import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import mask_to_index

from training_tools import Logger, set_seed
from graph_tools import get_features
from model import AFGNN



def data_loader(mask, batch_size, shuffle):
    idx = mask_to_index(mask)
    loader = DataLoader(idx, batch_size=batch_size, shuffle=shuffle)

    return loader


def train(train_loader, fea, mean_fea, true_y, model, optimizer, device):
    model.train()

    total_loss = 0.
    for batch in train_loader:
        batch_fea = fea[batch].to(device)
        batch_y = true_y[batch].to(device)
        mean_fea = mean_fea.to(device)
        optimizer.zero_grad()
        out = model(batch_fea, mean_fea)
        out = torch.log_softmax(out, dim=-1)
        loss = F.nll_loss(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


@torch.no_grad()
def test(fea, mean_fea, true_y, loader, model, device):
    model.eval()

    total_correct, total_num = 0., 0.

    for batch in loader:
        out = model(fea[batch].to(device), mean_fea.to(device))
        pred_y = out.argmax(dim=-1)
        total_correct += int((pred_y == true_y[batch].to(device)).sum())
        total_num += len(batch)

    acc= total_correct / total_num

    return acc


@torch.no_grad()
def attention_scores(fea, mean_fea, model, device):
    model.eval()
    scores = model.attention_scores(fea.to(device), mean_fea.to(device)).cpu()

    return scores


def run_experiments(data, kl, km, kh, num_heads, att_dropout, hid_dim, out_dim, pred_layers, dropout,
                    adaptive, runs, lr, weight_decay, batch_size, epochs, eval_steps, cuda, save_scores):
    set_seed()
    device = torch.device(f'cuda:{cuda}')

    masks = [data.train_mask, data.val_mask, data.test_mask]
    [print(torch.sum(mask)) for mask in masks]

    fea = get_features(data, kl, km, kh)
    mean_fea = fea.mean(dim=1, keepdim=True)

    model = AFGNN(data.num_features, 1+kl+km+kh, num_heads, att_dropout, hid_dim, out_dim,
                  pred_layers, dropout, adaptive).to(device)

    logger = Logger(runs)
    for run in range(runs):
        print('============================================')
        model.reset_parameters()

        optimizer = torch.optim.AdamW(model.parameters(), lr =lr, weight_decay=weight_decay)

        train_loader = data_loader(masks[0], batch_size, shuffle=True)
        val_loader = data_loader(masks[1], batch_size, shuffle=False)
        test_loader = data_loader(masks[2], batch_size, shuffle=False)

        for epoch in range(1, 1 + epochs):
            loss = train(train_loader, fea, mean_fea, data.y, model, optimizer, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

            if epoch > 39 and epoch % eval_steps == 0:
                train_acc = test(fea, mean_fea, data.y, train_loader, model, device)
                val_acc = test(fea, mean_fea, data.y, val_loader, model, device)
                test_acc = test(fea, mean_fea, data.y, test_loader, model, device)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                result = train_acc, val_acc, test_acc
                logger.add_result(run, result)

        if save_scores:
            torch.save(attention_scores(fea, mean_fea, model, device), f'scores_{run+1}.pt')

        logger.print_statistics(run)
    logger.print_statistics()
