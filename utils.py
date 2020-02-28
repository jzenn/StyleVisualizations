import torch

import torchvision.utils as utils


def compute_i_th_moment(input, i):
    """
    computes the i-th moment of the input tensor channel-wise
    :param input: tensor of size (n, c, h, w), n=1
    :param i: the moment one wants to compute
    :return: tensor with the i-th moment of every channel
    """
    # get the input size
    input_size = input.size()

    # (n, c, h, w)
    n = input_size[0]
    c = input_size[1]

    mean = torch.mean(input.view(n, c, -1), dim=2, keepdim=True).view(n, c, 1, 1)

    eps = 1e-5
    var = torch.var(input.view(n, c, -1), dim=2, keepdim=True) + eps
    std = torch.sqrt(var).view(n, c, 1, 1)

    if i == 1:
        return mean
    elif i == 2:
        return std
    else:
        return torch.mean((((input - mean) / std).pow(i)).view(n, c, -1), dim=2, keepdim=True).view(n, c, 1, 1)


def gram_matrix(input):
    """ gets the gram matrix of the input and normalizes it """
    n, c, h, w = input.size()
    features = input.view(n * c, h * w)
    gram = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return gram.div(n * c * h * w)


def split_even_odd(x):
    """
    split a list into two different lists by the even and odd entries
    :param x: the list
    :return: two lists with even and odd entries of x respectively
    """
    n, c, h, w = x.size()
    x = x.view(n * c, -1)
    return [x[:, 0:((h * w) - ((h * w) % 2))].view(n * c, -1, 2)[:, :, 1],
            x[:, 0:((h * w) - ((h * w) % 2))].view(n * c, -1, 2)[:, :, 0]]


def gaussian_kernel(x, y, alpha):
    """
    compute a Gaussian kernel for vector x and y
    :param x: data list
    :param y: data list
    :param alpha: parameter for the Gaussian kernel
    :return: the Gaussian kernel
    """
    # e^(-a * |x - y|^2)
    return torch.exp(-alpha * (x - y).pow(2))


def h(x_i, y_i, x_j, y_j, alpha):
    """
    helper function for the MMD O(n) computation
    :param x_i: odd entries of x
    :param y_i: odd entries of y
    :param x_j: even entries of x
    :param y_j: even entries of y
    :param alpha: the parameter for the Gaussian kernel
    :return: the value for the Gaussian kernel
    """
    # compute kernel values
    s1 = gaussian_kernel(x_i, x_j, alpha)
    s2 = gaussian_kernel(y_i, y_j, alpha)
    s3 = gaussian_kernel(x_i, y_j, alpha)
    s4 = gaussian_kernel(x_j, y_i, alpha)

    # return result of h
    return s1 + s2 - s3 - s4


def linear_time_mmd(x, y, alpha=1/10):
    """
    compute the linear time O(n) approximation of the MMD
    :param x:
    :param y:
    :param alpha:
    :return:
    """
    # split tensors x and y channel-wise based on its index
    x_even, x_odd = split_even_odd(x)
    y_even, y_odd = split_even_odd(y)

    # number of even/odd elements
    # _, result_length = x_even.size()

    # return mmd approximation
    return torch.abs(torch.sum(h(x_odd, y_odd, x_even, y_even, alpha)))


def save_layer_images(configuration, images, style_loss_number, image_number):
    """
    save the images at a certain layer next to each other
    :param configuration: the config file
    :param images: the images
    :param style_loss_number: number of the style loss module
    :param image_number: number of the image
    :return:
    """
    if style_loss_number == 1:
        image_saving_path = configuration['image_saving_path_mean']
    elif style_loss_number == 2:
        image_saving_path = configuration['image_saving_path_mean_std']
    elif style_loss_number == 3:
        image_saving_path = configuration['image_saving_path_mean_std_skew']
    elif style_loss_number == 4:
        image_saving_path = configuration['image_saving_path_mean_std_skew_kurtosis']
    elif style_loss_number == 5:
        image_saving_path = configuration['image_saving_path_gram']
    elif style_loss_number == 6:
        image_saving_path = configuration['image_saving_path_mmd']
    else:
        raise RuntimeError('unrecognized style loss number: {}'.format(style_loss_number))

    utils.save_image(images, filename='{}/{}_style_image_{}_balancing.jpeg'.format(
        image_saving_path,
        configuration['image_folder'],
        image_number), nrow=6, pad_value=1)


def save_all_images(configuration, images, number_style_images):
    """
    save all images ordered by their loss at different layers
    :param configuration: the config file
    :param images: the images
    :param number_style_images: number of style images available
    :return:
    """
    # iterate over all layer responses
    for j in range(1, 5):
        print('catching all images in layer response {}'.format(j))
        # iterate over all style images
        for i in range(number_style_images):
            print('catching style image {} in layer response {}'.format(i, j))
            # collect the layer responses of one style image
            layer_responses = [images[i][0]]
            # collect the different moment responses
            for k in range(5):
                print('catching moment response for moment {}'.format(k))
                style_image = k * number_style_images + i
                layer_responses += [images[style_image][j]]
            print('saving the current image stack')
            utils.save_image(layer_responses, filename='{}/{}_layer_{}_style_image_{}_balancing.jpeg'.format(
                configuration['image_saving_path_moment_comparison'],
                configuration['image_folder'],
                j, i), nrow=6, pad_value=1)
