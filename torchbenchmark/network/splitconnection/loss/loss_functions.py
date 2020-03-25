import numpy as np
import torch as th
from torch import Tensor

def softmax_crossentropy_with_logits(logits: Tensor, reference_answers: Tensor):
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]

    xentropy = - logits_for_answers + th.log(th.sum(th.exp(logits), axis=-1))

    return xentropy


def grad_softmax_crossentropy_with_logits(logits: Tensor, reference_answers: Tensor):
    """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""
    ones_for_answers = th.zeros_like(logits)
    ones_for_answers[th.arange(len(logits)), reference_answers] = 1

    softmax = th.exp(logits) / th.exp(logits).sum(axis=-1, keepdims=True)

    return (- ones_for_answers + softmax) / logits.shape[0]
