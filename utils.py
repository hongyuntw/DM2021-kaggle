from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn


emotion2idx = {'anticipation': 0, 'sadness': 1, 'fear': 2, 'joy': 3, 'anger': 4, 'trust': 5, 'disgust': 6, 'surprise': 7}
idx2emotion = {0: 'anticipation', 1: 'sadness', 2: 'fear', 3: 'joy', 4: 'anger', 5: 'trust', 6: 'disgust', 7: 'surprise'}


custom_class_weight = [10000/248935, 10000/193437, 10000/63999, 10000/516017, 10000/39867, 10000/205478, 10000/139101, 10000/48729]

    


def compute_acc(pred, label):
    pred = pred.detach()
    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(dim=-1)
    return (pred == label).float().mean()