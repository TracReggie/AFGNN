import argparse
import torch

from prepare_data import return_dataset
from graph_tools import get_fea_list
from training_tools import set_seed
from experiments import run_experiments
from model import AFGNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', type=str, default='test_model')
    parser.add_argument('--dataset_name', type=str, default="cora")
    parser.add_argument('--save_scores', type=bool, default=False)

    parser.add_argument('--kl', type=int, default=5)
    parser.add_argument('--km', type=int, default=3)
    parser.add_argument('--kh', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--adaptive', type=bool, default=True)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--att_dropout', type=float, default=0.5)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()
    print(args)

    set_seed()

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    data, dataset = return_dataset(args.dataset_name)

    masks = [data.train_mask, data.val_mask, data.test_mask]
    [print(torch.sum(mask)) for mask in masks]

    fea_list = get_fea_list(data, args.kl, args.km, args.kh)

    model = AFGNN(data.num_features, len(fea_list), args.num_heads, args.att_dropout, args.hid_dim,
                  dataset.num_classes, args.num_layers, args.dropout, args.adaptive).to(device)

    run_experiments(args, model, masks, fea_list, data.y, device, args.save_scores)
    print(args)


if __name__ == "__main__":
    main()
