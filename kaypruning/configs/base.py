from kaypruning.utils.utils import DotDict

data_configs = DotDict({
    "prefetch": 10,
    "repeat": 1,
    "shuffle": 5000,
})

model_hparams = DotDict({
    "hidden_layers_size": [1000, 1000, 500, 200],
    "num_features": 28 * 28,
    "num_labels": 10,
})

training_hparams = DotDict({
    "lr": 1e-3,
    "batch_size": 32,  # 256,
    "path": "/tmp/kay/models/kaypruning/"
})

prune_configs = DotDict({
    "k": [0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99],
    "type": ["weight", "unit"]
})
