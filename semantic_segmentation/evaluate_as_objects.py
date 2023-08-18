import os
import argparse
import geopandas as gpd
import pandas as pd

# example on how to run
# python /workspace/code/semantic_segmentation/evlaute_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/minimal_curvature/ /workspace/data/logfiles/ExceptionUNet/05m/minimal_curvature1/test_objects.csv


def object_evaluation(label_dir, polygon_dir):
    TP_list = []
    FN_list = []
    FP_list = []
    
    # TP and FN
    for label in os.listdir(label_dir):
        if label.endswith('.shp'):
            if os.path.isfile(polygon_dir + label):
                gt = gpd.read_file(label_dir + label)
                pred = gpd.read_file(polygon_dir + label)
                if pred.empty == True:
                    FN_list.append(len(gt))
                else:        
                    TP = gt.intersects(pred.unary_union)
                    for item in TP:
                        if item == True:
                            TP_list.append(1)
                        elif item == False:
                            FN_list.append(1)

    # FP             
    for prediction in os.listdir(polygon_dir):
        if prediction.endswith('.shp') and os.path.isfile(label_dir + prediction):
            gt = gpd.read_file(label_dir + prediction) # read label polygon         
            pred = gpd.read_file(polygon_dir + prediction) # read predicted polygon
            #gt.to_crs(3006)
            #pred.to_crs(3006)
            intersect = pred.intersects(gt.unary_union)
            for item in intersect:
                if item == False:
                    FP_list.append(1)
                    
    TP = sum(TP_list)
    FP = sum(FP_list)
    FN = sum(FN_list)  
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    return(TP, FP, FN, Precision, Recall, F1)

def main(label_dir, inference_dir, out_csv):
    results = {'TP': [], 'FP': [], 'FN': [], 'Precision': [], 'Recall': [], 'F1': []}
    TP, FP, FN, Precision, Recall, F1 = object_evaluation(label_dir, inference_dir)
    results['TP'].append(TP)
    results['FP'].append(FP)
    results['FN'].append(FN)
    results['Precision'].append(Precision)
    results['Recall'].append(Recall)
    results['F1'].append(F1)
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print('evaluated ', inference_dir, 'results stored in ', out_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                       description='Evaluate model on given images',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('label_dir', help='Path to image folder')
    parser.add_argument('inference_dir', help='Path to inference polygons')
    parser.add_argument('out_csv', help='Path to output CSV file')

    args = vars(parser.parse_args())
    main(**args)