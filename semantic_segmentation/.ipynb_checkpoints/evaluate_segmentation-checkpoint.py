import os
import numpy as np
import pandas as pd
import sklearn.metrics
import tifffile

def perf_measure(gt, pred):
    pred = pred.flatten()
    gt = gt.flatten()
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(pred)): 
        if gt[i]==pred[i]==1:
           TP += 1
        if pred[i]==1 and gt[i]!=pred[i]:
           FP += 1
        if gt[i]==pred[i]==0:
           TN += 1
        if pred[i]==0 and gt[i]!=pred[i]:
           FN += 1

    return(TP, FP, TN, FN)

def evaluate(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    fmes = sklearn.metrics.f1_score(gt, pred)
    acc = sklearn.metrics.accuracy_score(gt, pred)
    rec = sklearn.metrics.recall_score(gt, pred)
    jacc = sklearn.metrics.jaccard_score(gt, pred)
    TP, FP, TN, FN = perf_measure(gt, pred)
    return fmes, acc, rec, jacc, TP, FP, TN, FN 
# raw
# python /workspace/code/semantic_segmentation/evaluate_segmentation.py /workspace/data/test_data_pits/inference/ /workspace/data/test_data_pits/labels/ /workspace/data/logfiles/pits/pits7/test_pred.csv
# post processed
# python /workspace/code/semantic_segmentation/evaluate_segmentation.py /workspace/data/test_data_pits/inference/ /workspace/data/test_data_pits/labels/ /workspace/data/logfiles/pits/pits7/test_pred_postprocessed.csv

def main(img_path, gt_path, out_csv):
    results = {'fmes': [], 'acc': [], 'rec': [], 'jacc': [], 'TP': [], 'FP': [], 'TN': [], 'FN': []}
    for i in os.listdir(img_path):
        if i.endswith('.tif'):
            img = tifffile.imread(img_path + i)
            gt = tifffile.imread(gt_path + i)


            fmes, acc, rec, jacc, TP, FP, TN, FN = evaluate(img, gt)
            results['fmes'].append(fmes)
            results['acc'].append(acc)
            results['rec'].append(rec)
            results['jacc'].append(jacc)
            results['TP'].append(TP)
            results['FP'].append(FP)
            results['TN'].append(TN)
            results['FN'].append(FN)

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Evaluate model on given images',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('img_path', help='Path to image folder')
    parser.add_argument('gt_path')
    parser.add_argument('out_csv', help='Path to output CSV file')

    args = vars(parser.parse_args())
    main(**args)