# neural-style-loss
Style similarity estimation between images utilizing [neural style transfer network](https://arxiv.org/abs/1508.06576). The code is based on [Tensorflow implementation](https://github.com/anishathalye/neural-style) of neural style transfer network.

This code simply calculates L2 loss of "style" Gram matrices to estimate style (dis)similarity between images without any training process.

# Usage
## Comparing two images
```python neural_style_loss.py --image1 <first image file> --image2 <second image file>```

Need [Pre-trained VGG network](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat) in the top level of this repository, or in the specified location via ```--network``` option

See ```python neural_style_loss.py --help``` for other options

## Comparing multiple images at once
```python style_loss_iterator.py --path <image file folder> --output <output csv file>```

Output table of normalized losses between images (JPEG/PNG) in the specified folder

# Requirements
- [Tensorflow](https://www.tensorflow.org)
- [Numpy](http://www.numpy.org)
- [Scipy](https://www.scipy.org)
- [Pillow](https://python-pillow.org)

# Reference
- [A Neural Algorithm of Artistic Style, L. Gatys et al. 2015](https://arxiv.org/abs/1508.06576)
- [Neural Style, A. Athalye 2015](https://github.com/anishathalye/neural-style)
