import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomNodeSplit

from experiments import run_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kl', type=int, default=5)
    parser.add_argument('--km', type=int, default=3)
    parser.add_argument('--kh', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--att_dropout', type=float, default=0.5)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--pred_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--adaptive', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--save_scores', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    T = RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2)
    dataset = Planetoid(root='/home/zq2/data', name='Cora', transform=T)
    data = dataset[0]

    run_experiments(data, args.kl, args.km, args.kh, args.num_heads, args.att_dropout, args.hid_dim,
                    dataset.num_classes, args.pred_layers, args.dropout, args.adaptive, args.runs, args.lr,
                    args.weight_decay, args.batch_size, args.epochs, args.eval_steps, args.cuda, args.save_scores)

    print('============================================')
    print(args)


if __name__ == "__main__":
    main()
