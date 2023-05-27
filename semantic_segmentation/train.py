import os
import logging
import tensorflow as tf

import utils.generator
import utils.unet
import utils.loss
import utils.metric


def write_dataset(selected, log_path, name):
    with open(os.path.join(log_path, name), 'w') as f:
        for path in selected:
            f.write('{}\n'.format(path))


def parse_weighting_mode(weighting):
    '''Convert given string into weighting mode representation
       Supported modes are: "auto", "mfb", "none" and comma separated weights

    Parameters
    ----------
    weighting : String indicating weighting mode

    Returns
    -------
    Generator weighting mode representation ("auto"|"mfb"|None|list)
    Model weighting mode representation (NONE|MANUAL|FOCAL)

    '''
    generator = None
    model = None
    if weighting.lower() == 'auto':
        generator = 'auto'
        model = utils.unet.SegmentationModelInterface.WEIGHTING.MANUAL
    elif weighting.lower() == 'mfb':
        generator = 'mfb'
        model = utils.unet.SegmentationModelInterface.WEIGHTING.MANUAL
    elif weighting.lower() == 'none':
        generator = None
        model = utils.unet.SegmentationModelInterface.WEIGHTING.NONE
    elif weighting.lower() == 'focal':
        generator = None
        model = utils.unet.SegmentationModelInterface.WEIGHTING.FOCAL
    else:
        generator = [float(f) for f in weighting.split(',')]
        model = utils.unet.SegmentationModelInterface.WEIGHTING.MANUAL

    return generator, model

def main(img_path, gt_path, log_path, model_type, seed, epochs, depth,
         batch_size, steps_per_epoch, band_wise, classes, weighting,
         model_path):

    # setup logging
    logging.basicConfig(
                filename=os.path.join(log_path, 'train.log'),
                filemode='w', level=logging.INFO,
                datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s %(name)s - %(levelname)s - %(message)s')

    # convert string representations of weights into lists
    generator_weighting, model_weighting = parse_weighting_mode(weighting)

    # set seed for tensorflow (and everything else except data generators,
    # which use their own random number gnerators, which are seeded separately)
    # does not work for some cuDNN operations - so possibly not totally
    # deterministic
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    model_cls = utils.unet.MODELS[model_type]
    train_gen = utils.generator.DataGenerator(
                                       img_path, gt_path, classes,
                                       seed=seed, size=0.8,
                                       steps_per_epoch=steps_per_epoch,
                                       augment=True,
                                       class_weights=generator_weighting,
                                       batch_size=batch_size,
                                       channel_last=model_cls.CHANNEL_LAST)
    valid_gen = utils.generator.DataGenerator(
                                       img_path, gt_path, classes,
                                       seed=seed,
                                       exclude=train_gen.selected,
                                       steps_per_epoch=steps_per_epoch,
                                       augment=False,
                                       flatten=generator_weighting is not None,
                                       channel_last=model_cls.CHANNEL_LAST)

    write_dataset(train_gen.selected, log_path, 'train_imgs.txt')
    write_dataset(valid_gen.selected, log_path, 'valid_imgs.txt')

    # enable entry block only in default unet mode
    model = model_cls(train_gen.input_shape, depth=depth,
                      classes=train_gen.class_num, entry_block=not band_wise,
                      weighting=model_weighting)
    assert isinstance(model, utils.unet.SegmentationModelInterface)

    if model_path is not None:
        logging.info('loaded weights from %s', model_path)
        model.load_weights(model_path)

    logging.info('start training')
    model.train(epochs, train_gen, valid_gen, log_path)
    logging.info('end training')


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
    parser.add_argument('model_type', help='Segmentation model to use',
                        choices=list(utils.unet.MODELS.keys()))
    parser.add_argument('--seed', help='Random seed', default=None, type=int)
    parser.add_argument('--depth', help='Depth of the used network',
                        default=None, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--steps_per_epoch', default=None, type=int)
    parser.add_argument('--band_wise', action='store_true',
                        help='Apply separable convolutions on input bands.')
    parser.add_argument('--batch_size', help='Number of patches per batch',
                        type=int, default=4)
    parser.add_argument('--classes', help='List of class labels in ground '
                        'truth - order needs to correspond to weighting order',
                        default='0,1,2')
    parser.add_argument('--weighting', help='Configure class weights - can be '
                        '"auto", "mfb", "none", "focal" or defined weight '
                        'string, e.g., "0.1,1,1"',
                        default='0.1,1,1')
    parser.add_argument('--model_path', default=None, help='Path to '
                        'pre-trained model')

    args = vars(parser.parse_args())
    main(**args)