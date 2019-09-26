# Registry Style like for-ai/rl

_PRUNE = dict()


def register(fn):
    global _PRUNE
    _PRUNE[fn.__name__] = fn
    return fn


def get_prune_fn(register, weight, k):
    return _PRUNE[register](weight, k)
