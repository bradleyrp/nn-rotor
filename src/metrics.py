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


def calculate_metrics(pred, true):
        
    pairs_pred = find_split_indices(pred)
    pairs_true = find_split_indices(true)
    
    n_pred = len(pairs_pred)
    n_true = len(pairs_true)

    rows = []
    
    row_template = {
        "ref": False,
        "iou": None,
        "offset": None,
        "onset": None,
        "intersection": None,
        "union": None,
    }

    for pair_pred in pairs_pred:

        i1, i2 = pair_pred
        
        row = row_template.copy()
        
        intersection = -1

        for pair_true in pairs_true:

            j1, j2 = pair_true
            
            intersection = min(i2, j2) - max(i1, j1)
            
            if intersection > 0:
                break
                
        if intersection < 0:
            continue
            rows.append(row_template)
            
        union = max(i2, j2) - min(j1, i1)
        
        row["ref"] = True
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
