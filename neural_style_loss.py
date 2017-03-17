import os
import tensorflow as tf
import numpy as np
import scipy.misc
from argparse import ArgumentParser
from sys import stderr
from PIL import Image

import vgg

# default arguments
LAYER_WEIGHT_EXP = 1
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--image1',
            dest='image1', help='first image',
            metavar='IMAGE1', required=True)
    parser.add_argument('--image2',
            dest='image2', help='second image',
            metavar='IMAGE2', required=True)
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--style-layer-weight-exp', type=float,
            dest='layer_weight_exp', help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
            metavar='LAYER_WEIGHT_EXP', default=LAYER_WEIGHT_EXP)
    parser.add_argument('--pooling',
            dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
            metavar='POOLING', default=POOLING)
    return parser

LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

try:
    reduce
except NameError:
    from functools import reduce

def styleloss(network, image1, image2, layer_weight_exp, pooling):
    """
    Calculate style similarity utilizing style (gram) matrix.
    This function returns "style loss", which indicates how dissimilar two input images are.
    """
    image1_shape = (1,) + image1.shape # (1, height, width, number)
    image2_shape = (1,) + image2.shape
    image1_features = {}
    image2_features = {}

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    layer_weight = 1.0
    layers_weights = {}
    for layer in LAYERS:
        layers_weights[layer] = layer_weight
        layer_weight *= layer_weight_exp

    # normalize layer weights
    layer_weights_sum = 0
    for layer in LAYERS:
        layer_weights_sum += layers_weights[layer]
    for layer in LAYERS:
        layers_weights[layer] /= layer_weights_sum

    # compute image1 features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=image1_shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        image1_pre = np.array([vgg.preprocess(image1, vgg_mean_pixel)])
        for layer in LAYERS:
            features = net[layer].eval(feed_dict={image: image1_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features)
            image1_features[layer] = gram

    # compute image2 features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=image2_shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        image2_pre = np.array([vgg.preprocess(image2, vgg_mean_pixel)])
        for layer in LAYERS:
            features = net[layer].eval(feed_dict={image: image2_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features)
            image2_features[layer] = gram

    # calculate style loss from gram matrices
    with tf.Graph().as_default():
        style_loss = 0
        style_losses = []
        for layer in LAYERS:
            temp_layer = net[layer]
            _, height, width, number = map(lambda i: i.value, temp_layer.get_shape())
            size = height * width * number
            image1_gram = image1_features[layer]
            image2_gram = image2_features[layer]
            style_losses.append(layers_weights[layer] * tf.nn.l2_loss(image1_gram - image2_gram) / size**2)
        style_losses = reduce(tf.add, style_losses)
        with tf.Session() as sess:
            style_loss = style_losses.eval()

        return style_loss

def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img

def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    image1 = imread(options.image1)
    image2 = imread(options.image2)
    if image1.shape[1] < image2.shape[1]:
        image2 = scipy.misc.imresize(image2, image1.shape[1] / image2.shape[1])
    else:
        image1 = scipy.misc.imresize(image1, image2.shape[1] / image1.shape[1])

    style_loss =  styleloss(
        network=options.network,
        image1=image1,
        image2=image2,
        layer_weight_exp=options.layer_weight_exp,
        pooling=options.pooling
    )

    print('style_loss: '+str(style_loss))


if __name__ == '__main__':
    main()
