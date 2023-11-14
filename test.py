import argparse

from experiments import run_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="cora")
    parser.add_argument('--save_scores', type=bool, default=False)

    parser.add_argument('--kl', type=int, default=10)
    parser.add_argument('--km', type=int, default=10)
    parser.add_argument('--kh', type=int, default=10)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--adaptive', type=bool, default=True)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--att_dropout', type=float, default=0.5)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()
    print(args)

    run_experiments(args)
    print(args)


if __name__ == "__main__":
    main()
