import numpy as np
from PIL import Image
import urllib2
from os.path import join, dirname
from cStringIO import StringIO
import caffe


caffe_path = '/home/ubuntu/caffe'
model_fn = join(caffe_path, 'ResNet-101-model.caffemodel')
proto_fn = join(caffe_path, 'ResNet-101-deploy.prototxt')
mean_file = join(caffe_path, 'ResNet_mean.binaryproto')

blob = caffe.proto.caffe_pb2.BlobProto()
blob.ParseFromString(open(mean_file, 'rb').read())
mean = np.array(caffe.io.blobproto_to_array(blob))[0]

caffe.set_mode_gpu()
net = caffe.Net(proto_fn, model_fn, caffe.TEST)

with open(join(caffe_path, 'synset.txt')) as f:
    synset = np.array(f.read().split('\n'))

    
def crop_image(image):
    '''Crop an image to specific height and width.
    Args:
        image : PIL Image instance
    Returns:
        np.ndarray: a `self.w`x`self.h`x3 matrix with values ranged from 0 to 1.
    '''
    assert isinstance(image, Image.Image)
    if len(image.split()) == 4:  # convert RGBA to RGB
        image = Image.merge('RGB', image.split()[:3])
    elif len(image.split()) == 1:  # convert black and white to RGB
        image = Image.merge('RGB', image.split()[0:1] * 3)

    aspect = float(image.size[0]) / float(image.size[1])
    target_aspect = 1.0
    if aspect > target_aspect:  # width should be decreased
        target_width = image.size[1] * target_aspect
        pad = (image.size[0] - target_width) // 2
        box = (pad, 0, pad + target_width, image.size[1])
        image = image.crop(box)
    elif aspect < target_aspect:  # height should be decreased
        target_height = image.size[0] / target_aspect
        pad = (image.size[1] - target_height) // 2
        box = (0, pad, image.size[0], pad + target_height)
        image = image.crop(box)

    image = np.array(image.resize((224, 224), Image.BICUBIC))
    img_array = image.transpose((2, 0, 1))[[2, 1, 0]] - mean
    return img_array


def load_image(x):
    if x.lower().startswith('http') or x.lower().startswith('ftp'):
        image = Image.open(StringIO(urllib2.urlopen(x).read()))
    else:
        image = Image.open(x)
    return image


def preprocess(f):
    '''Preprocess an image for neural networks input. The image will be
        croped to 224x224x3, then transposed to 3x224x224. The sequence of
        colors is B/G/R for Caffe input.
    Args:
        f (str): the url or path of the image.
    Returns:
        np.ndarray: 3x224x224 matrix.
    '''
    if not isinstance(f, list):
        f = [f]

    tmp = []
    for img in f:
        if isinstance(img, basestring):
            image = load_image(img)
        elif isinstance(img, Image.Image):
            image = img
        else:
            raise TypeError('img type is not support')

        tmp.append(crop_image(image))
    return tmp


def check_data(img_data):
    if isinstance(img_data, (basestring, Image.Image)):
        img_data = np.array(preprocess(img_data))
    elif isinstance(img_data, list) and isinstance(img_data[0], (basestring, Image.Image)):
        img_data = np.array(preprocess(img_data))
    elif isinstance(img_data, list) and len(img_data[0].shape) == 3:
        img_data = np.array(img_data)
    elif len(img_data.shape) == 3:
        img_data = np.array([img_data])
    else:
        raise ValueError('Input img_data format is wrong')
    return img_data
