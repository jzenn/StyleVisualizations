import torch
import torchvision.transforms as transforms

import os

from PIL import Image

# the device being on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# size of the images
imsize = 256

# loading the test images to visualize them
loader = transforms.Compose([
    # scale images / resize images
    transforms.Resize(imsize),
    # crop center regions of size 256
    transforms.CenterCrop(256),
    # transform the image into a torch tensor
    transforms.ToTensor()
])

# un-loading the test images to visualize them
un_loader = transforms.ToPILImage()  # reconvert into PIL image


def get_images(image_path):
    """
    get all images at the specified image_path (! no check for actual image files)
    :param image_path: the path that is searched
    :return: number of images, file paths
    """
    images = os.listdir(image_path)
    number_images = len(images)
    image_file_paths = ['{}/{}'.format(image_path, images[i]) for i in range(number_images)]
    return number_images, image_file_paths


def load_image(image_path):
    """
    loads an image from the specified image path
    :param image_path:
    :return: the loaded image
    """
    image = Image.open(image_path)
    image = loader(image)
    image = image.unsqueeze(0)
    image = image.to(device, torch.float)
    return image
