import matplotlib.pyplot as plt


def plot(x=[0, 25, 50, 60, 70, 80, 90, 95, 97, 99], y=0, prune_type='weight', type='Accuracy'):
    plt.plot(x, y)
    plt.title(f'{prune_type} Pruning')
    plt.xlabel('Sparsity (%)')
    plt.ylabel(f'Test {type}')
    plt.show()
