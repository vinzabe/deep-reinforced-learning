"""Sample loader shipped with the model -- intentionally insecure."""
import pickle
import torch


def load(path):
    # bad: torch.load() goes through pickle by default
    state = torch.load(path)
    return state


def eval_expr(s):
    # bad: eval on untrusted input
    return eval(s)


def manual(path):
    with open(path, "rb") as fh:
        return pickle.loads(fh.read())
