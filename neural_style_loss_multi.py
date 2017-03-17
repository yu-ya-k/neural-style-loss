import os
import scipy.misc
import pandas as pd
from argparse import ArgumentParser
import time

from neural_style_loss import styleloss, imread

# default arguments
OUTPUT = 'output.csv'
LAYER_WEIGHT_EXP = 1
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'
NORMALIZE = 1
VERBOSE = 1
TIMEIT = 1

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--path',
            dest='path', help='path to image folder',
            metavar='PATH', required=True)
    parser.add_argument('--output',
            dest='output', help='output path (default %(default)s)',
            metavar='OUTPUT', default=OUTPUT)
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--style-layer-weight-exp', type=float,
            dest='layer_weight_exp', help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
            metavar='LAYER_WEIGHT_EXP', default=LAYER_WEIGHT_EXP)
    parser.add_argument('--pooling',
            dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
            metavar='POOLING', default=POOLING)
    parser.add_argument('--normalize',
            dest='normalize', help='normalize output values (default %(default)s)',
            metavar='NORMALIZE', default=NORMALIZE)
    parser.add_argument('--verbose',
            dest='verbose', help='print raw style loss value in each iteration (default %(default)s)',
            metavar='VERBOSE', default=VERBOSE)
    parser.add_argument('--timeit',
            dest='timeit', help='calculate and print the calculation time (default %(default)s)',
            metavar='TIMEIT', default=TIMEIT)
    return parser

def main():
    start_time = time.time()
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    # take only JPEG or PNG files
    path = options.path
    images = [f for f in os.listdir(path) if (f.endswith(".jpg") | f.endswith(".png"))]


    df = pd.DataFrame(0, index=images, columns=images)

    for i, impath1 in enumerate(images):
        image1 = imread(os.path.join(path,impath1))
        for j, impath2 in enumerate(images):
            if i<j:
                image2 = imread(os.path.join(path,impath2))
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
                df.iloc[i,j] = style_loss
                
                if options.verbose == 1:
                    print('style_loss between '+str(impath1)+' and '+str(impath2)+': '+str(style_loss))

            elif i>j:
                df.iloc[i,j] = df.iloc[j,i]
            else:
                df.iloc[i,j] = 0
    # normalize data array
    if options.normalize == 1:
        maxval = df.values.max()
        df = df/maxval

    output = options.output
    df.to_csv(output)
    
    if options.timeit == 1:
        print("calculation time: %s seconds" % (time.time() - start_time))

if __name__ == '__main__':
    main()
