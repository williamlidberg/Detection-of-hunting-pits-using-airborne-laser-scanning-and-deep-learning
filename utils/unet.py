# implementation adapted from:
# https://keras.io/examples/vision/oxford_pets_image_segmentation/

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

#from crfrnn_layer import CrfRnnLayer # don't forget to add this to path: export PYTHONPATH=/home/william/Downloads/crfasrnn_keras-master/src/:$PYTHONPATH


class XceptionUNet(object):

    UNET_MODES = ['default', 'wo_entry']

    def __init__(self, input_shape, depth=None, activation='softmax',
                 classes=2, mode=UNET_MODES[0], first_core_filters=128):
        '''Initialize Xception Unet
        Parameters
        ----------
        input_shape : Shape of the input images
        depth : Number of downsampling and corresponding upsampling layers,
                optional
        activation : Activation function to use in the hidden layers, optional
        classes : Number of target classes, optional
        mode : UNet mode - currently supported 'default', 'wo_entry'
        first_core_filters : Number of filters to use in first downsampling
                             block - determines the filter sizes in all
                             subsequent layers, optional
        Returns
        -------
        Initialized model object
        '''
        self.input_shape = input_shape

        depth = 2 if depth is None else depth
        self.activation = activation
        self.classes = classes
        if mode == self.UNET_MODES[0]:
            # Process input image by a CNN before starting the
            # downsampling with its separated convolutions
            self.entry_block = True
        elif mode == self.UNET_MODES[1]:
            self.entry_block = False
        else:
            raise ValueError('Unsupported mode: {}'.format(mode))
        self.__set_depth(depth, first_core_filters)
        self.padding = self.__compute_padding(self.input_shape, depth, self.entry_block)
        self.model = self.__setup_model()

    def __pad(self, size, downsampling_steps):
        div, rest = divmod(size, 2**downsampling_steps)
        if rest == 0:
            return (0, 0)
        else:
            padded = 2**downsampling_steps * (div + 1)
            padding = padded - size
            a = padding // 2
            b = padding - a
            return (a, b)

    def __compute_padding(self, input_shape, depth, entry_block):
        downsampling_steps = depth
        if entry_block:
            downsampling_steps += 1
        x, y, _ = input_shape
        l_r = self.__pad(x, downsampling_steps)
        t_b = self.__pad(y, downsampling_steps)

        return t_b, l_r
        

    def __set_depth(self, depth, first_core_filters):
        # setup filter list for downsampling
        start = np.log2(first_core_filters)
        start = int(start)
        self.down_sample = [2**i for i in range(start, start+depth)]
        # for deeper networks, reduce number of kernels to fit model into GPU
        # memory
        if depth >= 3:
            for i in range(2, len(self.down_sample)):
                self.down_sample[i] = self.down_sample[i] // 2

        # start downsampling with 32 filters if no CNN comes before the
        # downsampling block - keep configured depth
        if not self.entry_block:
            length = len(self.down_sample)
            self.down_sample.insert(0, 32)
            self.down_sample = self.down_sample[:length]

        # setup filter list for upsampling
        self.up_sample = self.down_sample.copy()
        self.up_sample.reverse()

        # add one more upsampling layer to compensate for initial CNN before
        # downsampling block
        if self.entry_block:
            self.up_sample.append(32)

    def __setup_model(self):
        inputs = keras.Input(shape=self.input_shape)

        # -- [First half of the network: downsampling inputs] -- #

        # add padding
        x = layers.ZeroPadding2D(padding=self.padding)(inputs)

        # Entry block
        if self.entry_block:
            x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in self.down_sample:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # -- [Second half of the network: upsampling inputs] -- #

        for filters in self.up_sample:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        # outputs = layers.Conv2D(self.num_classes, 3, activation="softmax",
        #                        padding="same")(x)
        outputs = layers.Conv2D(self.classes, 3, activation=self.activation,
                                padding="same")(x)
        # remove padding
        outputs = layers.Cropping2D(cropping=self.padding)(outputs)
        # reshape to make loss weighting possible
        outputs = layers.Reshape((-1, self.classes))(outputs)

        # Define the model
        model = keras.Model(inputs, outputs)
        return model


class XceptionUNetCRF(XceptionUNet):

    def __init__(self, input_shape, model_path=None, iterations=10):
        super().__init__(input_shape, activation=None)
        if model_path is not None:
            self.model.load_weights(model_path)
        self.iterations = iterations
        self.crf_model = self.__setup_crf_model()

    def __setup_crf_model(self):
        self.model.trainable = False

        outputs = layers.Reshape((self.input_shape[0],
                                  self.input_shape[1],
                                  self.classes))(self.model.outputs[0])
        # create fake RGB image
        inputs = layers.concatenate([self.model.inputs[0],
                                     self.model.inputs[0],
                                     self.model.inputs[0]], axis=3)
        crf_layer = CrfRnnLayer(image_dims=self.input_shape[:-1],
                                num_classes=self.classes,
                                theta_alpha=3.,
                                theta_beta=160.,
                                theta_gamma=3.,
                                num_iterations=self.iterations,
                                name='crfrnn')([outputs,
                                               inputs])
        # reshape to make loss weighting possible
        outputs = layers.Reshape((-1, self.classes))(crf_layer)
        # apply softmax
        outputs = layers.Softmax()(outputs)

        model = keras.Model(inputs=self.model.input,
                            outputs=outputs)
        return model


class UNet(object):

    def __init__(self, input_shape, depth=None):
        self.input_shape = input_shape

        depth = 4 if depth is None else depth
        self.__set_depth(depth)
        self.model = self.__setup_model()

    def __set_depth(self, depth):
        self.down_sample = [2**i for i in range(6, 6+depth)]
        #self.down_sample = [2**i for i in range(7, 7+depth)]
        #self.down_sample = [2**i for i in range(8, 8+depth)]
        self.up_sample = self.down_sample.copy()
        self.up_sample.reverse()

    def __setup_model(self):
        conv_layers = []
        inputs = keras.Input(self.input_shape)

        x = inputs
        # Down sampling
        for i, size in enumerate(self.down_sample):
            conv1 = layers.Conv2D(size, 3, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(x)
            conv1 = layers.Conv2D(size, 3, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(conv1)
            conv_layers.append(conv1)
            if i == (len(self.down_sample) - 1):
                conv1 = layers.Dropout(0.5)(conv1)
            x = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        # Middle
        size = self.down_sample[-1] * 2
        conv5 = layers.Conv2D(size, 3, activation='relu', padding='same',
                              kernel_initializer='he_normal')(x)
        conv5 = layers.Conv2D(size, 3, activation='relu', padding='same',
                              kernel_initializer='he_normal')(conv5)
        x = layers.Dropout(0.5)(conv5)

        # Up sampling
        for i, size in enumerate(self.up_sample):
            up6 = (layers.Conv2D(size, 2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')
                   (layers.UpSampling2D(size=(2, 2))(x)))

            merge6 = layers.concatenate([conv_layers.pop(), up6], axis=3)
            conv6 = layers.Conv2D(size, 3, activation='relu', padding='same',
                                  kernel_initializer='he_normal')(merge6)
            x = layers.Conv2D(size, 3, activation='relu', padding='same',
                              kernel_initializer='he_normal')(conv6)

        conv9 = layers.Conv2D(2, 3, activation='relu', padding='same',
                              kernel_initializer='he_normal')(x)
        conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

        model = keras.Model(inputs=inputs, outputs=conv10)
        return model