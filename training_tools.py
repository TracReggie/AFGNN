import random
import numpy as np
import os
import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            max_val = result[:, 1].max()
            mask = (result[:, 1] == max_val)
            print(f'Run {run + 1:02d}:')
            print(f'Highest Valid: {max_val:.2f}')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Final Train: {torch.max(result[:, 0][mask]):.2f}')
            print(f'Highest Test: {result[:, 2].max():.2f}')
            print(f'Final Test: {torch.max(result[:, 2][mask]):.2f}')
        else:
            results = 100 * torch.tensor(self.results)

            best_results = []
            for result in results:
                max_train = result[:, 0].max().item()
                max_test = result[:, 2].max().item()
                max_val = result[:, 1].max().item()
                mask = (result[:, 1] == max_val)
                final_train = torch.max(result[:, 0][mask]).item()
                final_test = torch.max(result[:, 2][mask]).item()
                best_results.append((max_train, max_test, max_val, final_train, final_test))

            best_result = torch.tensor(best_results)
            print('============================================')
            print(f'All runs:')
            result = best_result[:, 2]
            print(f'Highest Valid: {result.mean():.2f}±{result.std():.2f}')
            result = best_result[:, 0]
            print(f'Highest Train: {result.mean():.2f}±{result.std():.2f}')
            result = best_result[:, 3]
            print(f'Final Train: {result.mean():.2f}±{result.std():.2f}')
            result = best_result[:, 1]
            print(f'Highest Test: {result.mean():.2f}±{result.std():.2f}')
            result = best_result[:, 4]
            print(f'Final Test: {result.mean():.2f}±{result.std():.2f}')

            return result.mean()


def set_seed(seed=3407):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True  # 选择确定性算法
