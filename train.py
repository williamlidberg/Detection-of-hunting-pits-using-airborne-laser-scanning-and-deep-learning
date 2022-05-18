import utils.generator
import utils.unet
import utils.loss
import utils.metric
import tensorflow as tf
import numpy as np
import random

import os


def write_dataset(selected, log_path, name):
    with open(os.path.join(log_path, name), 'w') as f:
        for path in selected:
            f.write('{}\n'.format(path))


def parse_weighting_mode(weighting):
    '''Convert given string into weighting mode representation
       Supported modes are: "auto", "none" and comma separated weights
    Parameters
    ----------
    weighting : String indicating weighting mode
    Returns
    -------
    Weighting mode representation ("auto"|None|list)
    '''
    if weighting.lower() == 'auto':
        return 'auto'
    elif weighting.lower() == 'none':
        return None
    else:
        return [float(f) for f in weighting.split(',')]


def main(img_path, gt_path, log_path, seed, epochs, depth, batch_size,
         steps_per_epoch, unet_mode, classes, weighting):

    # convert string representations of class labels and weights into lists
    classes = [int(f) for f in classes.split(',')]
    weighting = parse_weighting_mode(weighting)

    # set seed for tensorflow (and everything else except data generators,
    # which use their own random number gnerators, which are seeded separately)
    # does not work for some cuDNN operations - so possibly not totally
    # deterministic
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    train_gen = utils.generator.DataGenerator(img_path, gt_path, classes,
                                              seed=seed, size=0.8,
                                              steps_per_epoch=steps_per_epoch,
                                              augment=True,
                                              class_weights=weighting,
                                              batch_size=batch_size)
    valid_gen = utils.generator.DataGenerator(img_path, gt_path, classes,
                                              seed=seed,
                                              exclude=train_gen.selected,
                                              steps_per_epoch=steps_per_epoch,
                                              augment=False)

    write_dataset(train_gen.selected, log_path, 'train_imgs.txt')
    write_dataset(valid_gen.selected, log_path, 'valid_imgs.txt')

    unet = utils.unet.XceptionUNet(train_gen.input_shape, depth=depth,
                                   classes=train_gen.class_num,
                                   mode=unet_mode)
    metrics = ['accuracy', tf.keras.metrics.Recall()]
    # record IoU for each class separately
#    for i in range(train_gen.class_num):
#        metrics.append(tf.keras.metrics.OneHotIoU(num_classes=train_gen.class_num,
#                                                  target_class_ids=[i,],
#                                                  name='{}_iou'.format(i)))
    unet.model.compile(
                       # optimizer="rmsprop",
                       optimizer="adam",
                       # optimizer=tf.keras.optimizers.SGD(momentum=0.9),
                       # loss=jaccard_distance_loss,
                       # loss='binary_crossentropy',
                       loss='categorical_crossentropy',
                       sample_weight_mode="temporal",
                       # loss=utils.loss.cross_entropy,
                       metrics=metrics)
    # utils.metric.f1_m, utils.metric.recall_m])
    # "categorical_crossentropy")

    callbacks = [
        # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10,
        #                                  mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10,
                                             min_lr=0.00001, mode='min'),
        tf.keras.callbacks.ModelCheckpoint(
                                        os.path.join(log_path, 'trained.h5'),
                                        monitor='val_loss',
                                        save_weights_only=True,
                                        verbose=0, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=5,
                                       write_grads=True, batch_size=2,
                                       write_images=True),
        tf.keras.callbacks.CSVLogger(os.path.join(log_path, 'log.csv'),
                                     append=True, separator=';')
    ]
    unet.model.fit_generator(train_gen, epochs=epochs, verbose=0,
                             callbacks=callbacks,
                             validation_data=valid_gen)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                        description='Train Model',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--img_path', action='append', help='Add path '
                        'to input images')
    parser.add_argument('gt_path', help='Path to groundtruth image folder')
    parser.add_argument('log_path', help='Path to folder where logging data '
                        'will be stored')
    parser.add_argument('--seed', help='Random seed', default=None, type=int)
    parser.add_argument('--depth', help='Depth of the used network',
                        default=None, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--steps_per_epoch', default=None, type=int)
    parser.add_argument('--unet_mode', choices=utils.unet.XceptionUNet.UNET_MODES,
                        default=utils.unet.XceptionUNet.UNET_MODES[0], 
                        help='Choose UNet architecture configuration')
    parser.add_argument('--batch_size', help='Number of patches per batch',
                        type=int, default=4)
    parser.add_argument('--classes', help='List of class labels in ground '
                        'truth - order needs to correspond to weighting order',
                        default='0,1')
    parser.add_argument('--weighting', help='Configure class weights - can be '
                        '"auto", "none" or defined weight string, e.g., '
                        '"0.1,1"',
                        default='0.1,1')

    args = vars(parser.parse_args())
    main(**args)