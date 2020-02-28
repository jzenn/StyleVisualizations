import torch

from net import get_vgg_model
from net import get_loss_module
from net import get_full_style_model
from net import get_input_optimizer

from data_loader import get_images
from data_loader import load_image

from utils import save_layer_images
from utils import save_all_images

# the device being on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(configuration):
    """
    the main training loop
    :param configuration: the config file
    :return:
    """
    image_path = configuration['image_path']
    print('using images from {}'.format(image_path))

    image_saving_path_mean = configuration['image_saving_path_mean']
    print('saving result images with mean loss to {}'.format(image_saving_path_mean))

    image_saving_path_mean_std = configuration['image_saving_path_mean_std']
    print('saving result images with mean + std loss to {}'.format(image_saving_path_mean_std))

    image_saving_path_mean_std_skew = configuration['image_saving_path_mean_std_skew']
    print('saving result images with mean + std + skew loss to {}'.format(image_saving_path_mean_std_skew))

    image_saving_path_mean_std_skew_kurtosis = configuration['image_saving_path_mean_std_skew_kurtosis']
    print('saving result images with mean + std + skew + kurtosis loss to {}'.format(image_saving_path_mean_std_skew_kurtosis))

    model_dir = configuration['model_dir']
    print('vgg-19 model dir is {}'.format(model_dir))

    vgg_model = get_vgg_model(configuration)
    print('got vgg model')

    number_style_images, style_image_file_paths = get_images(configuration)
    print('got {} style images'.format(number_style_images))

    all_images = []

    for i in range(1, 6):
        print('using style loss module: {}'.format(i))
        style_loss_module = get_loss_module(i)
        for j in range(number_style_images):
            print('computing image {} with style loss module {}'.format(j, i))
            style_image = load_image(style_image_file_paths[j])
            layer_images = [style_image.squeeze(0)]
            model, style_losses = get_full_style_model(configuration, vgg_model, style_image, style_loss_module)
            for k in range(4):
                print('computing loss at layer {}, image {}, loss module {}'.format(k, j, i))
                torch.manual_seed(13)
                image_noise = torch.randn(style_image.data.size()).to(device)
                layer_images += [train_full_style_model(model, style_losses, image_noise, k, i * (k+1) * 20000)
                                 .squeeze(0)]
            print('saving images at the different layers')
            save_layer_images(configuration, layer_images, i, j)
            all_images += [layer_images]
    print('saving the side-by-side comparisons')
    save_all_images(configuration, all_images, number_style_images)


def train_mmd(configuration):
    """
    training loop utilizing the MMD loss
    :param configuration: the config file
    :return:
    """
    image_path = configuration['image_path']
    print('using images from {}'.format(image_path))

    image_saving_path_mmd = configuration['image_saving_path_mmd']
    print('saving result images with mmd loss to {}'.format(image_saving_path_mmd))

    model_dir = configuration['model_dir']
    print('vgg-19 model dir is {}'.format(model_dir))

    vgg_model = get_vgg_model(configuration)
    print('got vgg model')

    number_style_images, style_image_file_paths = get_images(configuration)
    print('got {} style images'.format(number_style_images))

    i = 6

    print('using mmd loss module: {}'.format(i))
    style_loss_module = get_loss_module(i)
    for j in range(number_style_images):
        print('computing image {} with style loss module {}'.format(j, i))
        style_image = load_image(style_image_file_paths[j])
        layer_images = [style_image.squeeze(0)]
        model, style_losses = get_full_style_model(configuration, vgg_model, style_image, style_loss_module)
        for k in range(4):
            print('computing loss at layer {}, image {}, loss module {}'.format(k, j, i))
            torch.manual_seed(13)
            image_noise = torch.randn(style_image.data.size()).to(device)
            steps = (k+2) * 50000
            style_weight = 1000
            layer_images += [train_full_style_model(model, style_losses, image_noise, k,
                                                    steps, style_weight, early_stopping=True).squeeze(0)]
            # for debug
            save_layer_images(configuration, layer_images, i, j)
        print('saving images at the different layers')
        save_layer_images(configuration, layer_images, i, j)
    print('finished')


def train_gram(configuration):
    """
    training loop utilizing the Gram matrix loss
    :param configuration: the config file
    :return:
    """
    image_path = configuration['image_path']
    print('using images from {}'.format(image_path))

    image_saving_path_gram = configuration['image_saving_path_gram']
    print('saving result images with gram loss to {}'.format(image_saving_path_gram))

    model_dir = configuration['model_dir']
    print('vgg-19 model dir is {}'.format(model_dir))

    vgg_model = get_vgg_model(configuration)
    print('got vgg model')

    number_style_images, style_image_file_paths = get_images(configuration)
    print('got {} style images'.format(number_style_images))

    all_images = []
    i = 5

    print('using gram loss module: {}'.format(i))
    style_loss_module = get_loss_module(i)
    for j in range(number_style_images):
        print('computing image {} with style loss module {}'.format(j, i))
        style_image = load_image(style_image_file_paths[j])
        layer_images = [style_image.squeeze(0)]
        model, style_losses = get_full_style_model(configuration, vgg_model, style_image, style_loss_module)
        for k in range(4):
            print('computing loss at layer {}, image {}, loss module {}'.format(k, j, i))
            torch.manual_seed(13)
            image_noise = torch.randn(style_image.data.size()).to(device)
            steps = i * (k+1) * 20000
            layer_images += [train_full_style_model(model, style_losses, image_noise, k, steps).squeeze(0)]
        print('saving images at the different layers')
        save_layer_images(configuration, layer_images, i, j)
        all_images += [layer_images]
    print('finished')


def train_full_style_model(model, style_losses, image_noise, layer, steps, style_weight=100, early_stopping=True):
    """
    the actual training of the model
    :param model: the pre-trained VGG-19 model
    :param style_losses: a list of style losses to be inserted to the model
    :param image_noise: the noise image
    :param layer: the layer where the visualization is computed on
    :param steps: the number of steps to train
    :param style_weight: the weighting factor for the style loss term
    :param early_stopping: boolean determining whether to stop early if the loss drops below a certain threshold
    :return: the stylized image
    """
    optimizer = get_input_optimizer(image_noise)

    model.to(device)

    max_additional_iterations = 0

    loss = -1

    print('Optimizing.. {} steps'.format(steps))
    run = [0]
    while run[0] <= steps:
        def closure():
            # correct the values of updated input image
            image_noise.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(image_noise)

            loss = style_losses[layer].loss * style_weight

            loss.to(device)
            loss.backward(retain_graph=True)

            run[0] += 1

            if run[0] % 1000 == 0:
                print("run {} and {} to go".format(run[0], steps-run[0]))
                print('loss: {:4f}'.format(loss.item()))
                print()

            return loss

        loss = optimizer.step(closure)

        if early_stopping:
            if loss.item() > 1e-5 and run[0] == steps and max_additional_iterations < 450:
                print('training for another 1000 iterations')
                max_additional_iterations += 1
                steps += 1000
            if loss.item() < 1e-6:
                image_noise.data.clamp_(0, 1)
                print('saving image with loss: {}'.format(loss.item()))
                return image_noise

    image_noise.data.clamp_(0, 1)
    print('saving image with loss: {}'.format(loss.item()))
    return image_noise
