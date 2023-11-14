import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import mask_to_index

from training_tools import set_seed, Logger
from prepare_data import return_dataset
from graph_tools import get_fea_list
from model import AFGNN



def data_loader(mask, batch_size, shuffle):
    idx = mask_to_index(mask)
    loader = DataLoader(idx, batch_size=batch_size, shuffle=shuffle)

    return loader


def train(train_loader, fea_list, mean_list, true_y, model, optimizer, device):
    model.train()

    total_loss = 0.
    for batch in train_loader:
        train_list = [fea[batch].to(device) for fea in fea_list]
        train_y = true_y[batch].to(device)

        optimizer.zero_grad()
        out = model(train_list, mean_list)
        out = torch.softmax(out, dim=-1)

        loss = F.cross_entropy(out, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


@torch.no_grad()
def test(fea_list, mean_list, true_y, loader, model, device):
    model.eval()

    total_correct, total_num = 0., 0.

    for batch in loader:
        test_list = [fea[batch].to(device) for fea in fea_list]
        test_y = true_y[batch].to(device)
        out = model(test_list, mean_list)
        pred_y = out.argmax(dim=-1)

        total_correct += int((pred_y == test_y).sum())
        total_num += len(batch)

    acc= total_correct / total_num

    return acc


@torch.no_grad()
def attention_scores(fea_list, mean_list, model, device):
    model.eval()

    fea_list = [fea.to(device) for fea in fea_list]
    mean_list = [fea.to(device) for fea in mean_list]

    scores = model.attention_scores(fea_list, mean_list).cpu()

    return scores


def run_experiments(args):
    set_seed()
    data = return_dataset(f'{args.dataset_name}')
    device = torch.device(f'cuda:{args.cuda}')

    masks = [data.train_mask, data.val_mask, data.test_mask]
    [print(torch.sum(mask)) for mask in masks]

    fea_list = get_fea_list(data, args.kl, args.km, args.kh)

    model = AFGNN(data.num_features, len(fea_list), args.num_heads, args.att_dropout,
                  args.hid_dim, data.num_classes, args.num_layers, args.dropout, args.adaptive).to(device)

    logger = Logger(args.runs, args)
    for run in range(args.runs):
        print('============================================')
        model.reset_parameters()

        optimizer1 = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_loader = data_loader(masks[0], args.batch_size, shuffle=True)
        val_loader = data_loader(masks[1], args.batch_size, shuffle=False)
        test_loader = data_loader(masks[2], args.batch_size, shuffle=False)

        mean_list = [(torch.mean(fea, dim=0)).unsqueeze(0) for fea in fea_list]
        mean_list = [fea.to(device) for fea in mean_list]
        
        for epoch in range(1, 1 + args.epochs):
            loss = train(train_loader, fea_list, mean_list, data.y, model, optimizer1, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

            if epoch % args.eval_steps == 0:
                train_acc = test(fea_list, mean_list, data.y, train_loader, model, device)
                val_acc = test(fea_list, mean_list, data.y, val_loader, model, device)
                test_acc = test(fea_list, mean_list, data.y, test_loader, model, device)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                result = train_acc, val_acc, test_acc
                logger.add_result(run, result)
        
        if args.save_scores:
            torch.save(attention_scores(fea_list, mean_list, model, device), f'scores_{run+1}.pt')

        logger.print_statistics(run)
    logger.print_statistics()
    print('============================================')
