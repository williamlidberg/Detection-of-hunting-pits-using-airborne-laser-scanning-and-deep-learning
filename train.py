import utils.generator
import utils.unet
import utils.loss
import utils.metric
import tensorflow as tf

import os

UNET_MODES = ['default', 'wo_entry']


def write_dataset(selected, log_path, name):
    with open(os.path.join(log_path, name), 'w') as f:
        for path in selected:
            f.write('{}\n'.format(path))


def main(img_path, gt_path, log_path, seed, epochs, depth,
         steps_per_epoch, unet_mode):
    train_gen = utils.generator.DataGenerator(img_path, gt_path, [0, 1, 2], seed=seed,
                                              size=0.8,
                                              steps_per_epoch=steps_per_epoch,
                                              augment=True,
                                              class_weights=[0.1, 1, 1],
                                              batch_size=4)
    valid_gen = utils.generator.DataGenerator(img_path, gt_path, [0, 1, 2], seed=seed,
                                              exclude=train_gen.selected,
                                              steps_per_epoch=steps_per_epoch,
                                              augment=False)

    write_dataset(train_gen.selected, log_path, 'train_imgs.txt')
    write_dataset(valid_gen.selected, log_path, 'valid_imgs.txt')

    # enable entry block only in default unet mode
    entry_block = unet_mode == UNET_MODES[0]
    unet = utils.unet.XceptionUNet(train_gen.input_shape, depth=depth,
                                   classes=train_gen.class_num,
                                   entry_block=entry_block)
    unet.model.compile(
                       # optimizer="rmsprop",
                       optimizer="adam",
                       # optimizer=tf.keras.optimizers.SGD(momentum=0.9),
                       # loss=jaccard_distance_loss,
                       # loss='binary_crossentropy',
                       loss='categorical_crossentropy',
                       sample_weight_mode="temporal",
                       # loss=utils.loss.cross_entropy,
                       metrics=['accuracy', tf.keras.metrics.Recall(),
                                tf.keras.metrics.MeanIoU(
                                    num_classes=train_gen.class_num)])
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

    parser = argparse.ArgumentParser(description='Train Model')
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
    parser.add_argument('--unet_mode', choices=UNET_MODES,
                        default=UNET_MODES[0], help='Choose UNet architecture'
                        'configuration')

    args = vars(parser.parse_args())
    main(**args)