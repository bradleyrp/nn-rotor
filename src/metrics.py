import torch
import numpy as np
import pandas as pd

σ = torch.sigmoid


def accuracy(pred, true):
    
    pred = pred.cpu()
    true = true.cpu()
        
    with torch.no_grad():
        pred = σ(pred) > 0.5
        
    score = torch.mean(((pred > 0.5) == (true > 0.5)).float())    
    
    return score.item()


def find_split_indices(a):
    pairs = []
    i1 = 0
    label = a[i1]
    for i, x in enumerate(a[1:]):

        if label != a[i]:
            if label == 1:
                pair = [i1, i]
                pairs.append(pair)
            i1 = i
            label = a[i]

    if label == 1:
        pair = [i1, len(a)]
        pairs.append(pair)
    
    return pairs


def calculate_metrics(a, b):
    
    pairs_a = find_split_indices(a)
    pairs_b = find_split_indices(b)
    
    na = len(pairs_a)
    nb = len(pairs_b)

    rows = []
    
    row_template = {
        "ref": False,
        "iou": None,
        "offset": None,
        "onset": None,
        "intersection": None,
        "union": None,
    }

    for pair_a in pairs_a:

        i1, i2 = pair_a

        match = True
        
        row = row_template.copy()

        for pair_b in pairs_b:

            j1, j2 = pair_b

            if j2 < i1:
                continue

            elif j1 > i2:
                match = False
                break

            break

        if (not match) or (nb == 0):
            rows.append(row_template)
            continue

        row["ref"] = match

        intersection = min(i2, j2) - max(i1, j1)
        union = max(i2, j2) - min(j1, i1)

        row["intersection"] = intersection
        row["union"] = union

        if intersection * union == 0:
            iou = 0
        else:
            iou = intersection / union      

        row["iou"] = iou

        onset = i1 - j1
        offset = i2 - j2

        row["onset"] = onset
        row["offset"] = offset

        rows.append(row)

    df = pd.DataFrame(rows)
    
    return df
