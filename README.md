# neural-style-loss
Style similarity estimation between images utilizing [neural style transfer network](https://arxiv.org/abs/1508.06576). The code is based on [Tensorflow implementation](https://github.com/anishathalye/neural-style) of neural style transfer network.

This code simply calculates squared loss of "style" Gram matrices from intermediate layers of CNN to estimate style (dis)similarity between images without any training process.

# Usage
## Comparing two images
```python neural_style_loss.py --image1 <first image file> --image2 <second image file>```

Need [Pre-trained VGG network](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat) in the top level of this repository, or in the specified location via ```--network``` option

See ```python neural_style_loss.py --help``` for other options

## Comparing multiple images at once
```python neural_style_loss_multi.py --path <image file folder> --output <output csv file>```

Output table of style losses between images (JPEG/PNG) in the specified folder

# Requirements
- [Tensorflow](https://www.tensorflow.org)
- [Numpy](http://www.numpy.org)
- [Scipy](https://www.scipy.org)
- [Pandas](http://pandas.pydata.org)
- [Pillow](https://python-pillow.org)

# Reference
- [A Neural Algorithm of Artistic Style, L. Gatys et al. 2015](https://arxiv.org/abs/1508.06576)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition, K. Simonyan and A. Zisserman 2014](https://arxiv.org/pdf/1409.1556.pdf)
- [Neural Style, A. Athalye 2015](https://github.com/anishathalye/neural-style)
