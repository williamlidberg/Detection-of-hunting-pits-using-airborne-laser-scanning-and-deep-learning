import numpy as np
import pandas as pd
import sklearn.metrics

import utils.generator
import utils.unet


def perf_measure(gt, pred):

    TP = np.sum(gt & pred)
    FP = np.sum((1 - gt) & pred)
    TN = np.sum((1 - gt) & (1 - pred))
    FN = np.sum(gt & (1 - pred))

    scores = [('{}_tp', TP), ('{}_fp', FP),
              ('{}_tn', TN), ('{}_fn', FN)]

    return scores


def calculate_metrics(results, pred, gt, i):
    pred = pred.flatten().astype(bool)
    gt = gt.flatten().astype(bool)

    metrics = [('{}_fmes', sklearn.metrics.f1_score),
               ('{}_acc', sklearn.metrics.accuracy_score),
               ('{}_rec', sklearn.metrics.recall_score),
               ('{}_jacc', sklearn.metrics.jaccard_score)]

    for name, func in metrics:
        results.setdefault(name.format(i), []).append(func(gt, pred))

    scores = perf_measure(gt, pred)
    for name, val in scores:
        results.setdefault(name.format(i), []).append(val)


def main(img_path, gt_path, selected_imgs, model_path, model_type, out_path,
         classes, band_wise, depth):
    size = 1 if selected_imgs is None else None
    model_cls = utils.unet.MODELS[model_type]
    valid_gen = utils.generator.DataGenerator(img_path, gt_path,
                                              include=selected_imgs,
                                              classes=classes,
                                              size=size, augment=False,
                                              channel_last=model_cls.CHANNEL_LAST)
    model = model_cls(valid_gen.input_shape, depth=depth,
                      classes=valid_gen.class_num, entry_block=not band_wise)
    model.load_weights(model_path)

    results = {}
    valid_it = iter(valid_gen)

    for img, gt in valid_it:
        out = model.proba(img)
        classes = out.shape[-1]

        # convert probabilities to labels
        class_pred = np.argmax(out, axis=-1)
        predicted = np.zeros(out.shape[1:])
        # set the entry corresponding to the class with the highest probability
        # to 1 - the rest is 0
        predicted[list(range(len(predicted))), class_pred] = 1

        # fix groundtruth if needed
        if gt.shape[-1] != classes:
            gt = np.swapaxes(gt, 1, 2)

        for i in range(classes):
            calculate_metrics(results, predicted[:, i], gt[0, :, i], i)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Evaluate model on given images',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--img_path', action='append', help='Add path '
                        'to input images')
    parser.add_argument('gt_path')
    parser.add_argument('model_path')
    parser.add_argument('model_type', help='Segmentation model to use',
                        choices=list(utils.unet.MODELS.keys()))
    parser.add_argument('out_path', help='Path to output CSV file')
    parser.add_argument('--selected_imgs', help='Path to CSV file with '
                        'selected image indices', default=None)
    parser.add_argument('--classes', help='List of class labels in ground '
                        'truth - order needs to correspond to weighting order',
                        default='0,1')
    parser.add_argument('--band_wise', action='store_true',
                        help='Apply separable convolutions on input bands.')
    parser.add_argument('--depth', type=int, default=4)

    args = vars(parser.parse_args())
    main(**args)