import matplotlib.pyplot as plt

from kaypruning.configs import prune_configs


def plot(x=prune_configs.k, y=0, prune_type='weight', type='Accuracy'):
    plt.plot(x, y)
    plt.title(f'{prune_type} Pruning')
    plt.xlabel('Sparsity (%)')
    plt.ylabel(f'Test {type}')
    plt.show()
