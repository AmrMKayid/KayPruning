# Registry Style like for-ai/rl: https://github.com/for-ai/rl/blob/master/rl/models/registry.py

_MODELS = dict()


def register(fn):
    global _MODELS
    _MODELS[fn.__name__] = fn
    return fn


def get_model(register):
    return _MODELS[register]()
