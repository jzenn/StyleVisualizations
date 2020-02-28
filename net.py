import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.models as models

import copy

from utils import compute_i_th_moment
from utils import gram_matrix
from utils import linear_time_mmd

# the device being on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# normalization mean and std (of pre-trained PyTorch model)
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


def get_vgg_model(configuration):
    """
    get the pre-trained VGG-19 model from the PyTorch framework
    :param configuration: the config file
    :return: the pre-trained VGG-19 model
    """
    vgg = models.vgg19()
    vgg_pre_trained_state_dict = torch.utils.model_zoo.load_url(configuration['model_url'],
                                                                model_dir=configuration['model_dir'])
    vgg.load_state_dict(vgg_pre_trained_state_dict)
    vgg = vgg.features.to(device).eval()
    return vgg


def get_loss_module(i):
    """
    returns the loss module corresponding to i
    :param i:
    :return:
    """
    if i == 1:
        return StyleLossMean
    elif i == 2:
        return StyleLossMeanStd
    elif i == 3:
        return StyleLossMeanStdSkew
    elif i == 4:
        return StyleLossMeanStdSkewKurtosis
    elif i == 5:
        return StyleLossGramMatrix
    elif i == 6:
        return StyleLossMMD
    else:
        raise RuntimeError('could not recognize i = {}'.format(i))


def get_full_style_model(configuration, vgg_model, style_image, style_loss_module):
    """
    produce the full model from the pre-trained VGG-19 model
    :param configuration:
    :param vgg_model:
    :param style_image:
    :param style_loss_module:
    :return:
    """
    vgg_model = copy.deepcopy(vgg_model)

    print(vgg_model)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # the style losses
    style_losses = []

    # the layers the loss is computed upon
    layers = configuration['layers']

    model = nn.Sequential(normalization)

    i = 1
    j = 1
    for layer in vgg_model.children():
        if isinstance(layer, nn.Conv2d):
            name = 'conv_{}_{}'.format(i, j)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            j = 1
            i += 1
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}_{}'.format(i, j)
            # this inplace=False is very important, otherwise PyTorch throws an exception
            # ('one of the variables needed for gradient computation has been modified by an inplace operation')
            layer = nn.ReLU(inplace=False)
            j += 1
        else:
            raise RuntimeError('unrecognized layer')

        model.add_module(name, layer)

        if name in layers:
            # add style loss:
            target_feature = model(style_image).detach()
            style_loss = style_loss_module(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], style_loss_module):
            break

    # can totally trim of the model after the last style loss layer
    model = model[:(i + 1)]

    print(model)

    return model, style_losses


def get_input_optimizer(input_img):
    optimizer = optim.Adam([input_img.requires_grad_()])
    return optimizer


class Normalization(nn.Module):
    """
    normalization module to normalize the image data with mean and std
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class StyleLossMMD(nn.Module):
    """
    Style Loss MMD
    """
    def __init__(self, target_feature, alpha=1/10):
        super(StyleLossMMD, self).__init__()
        self.target = target_feature.detach()
        self.alpha = alpha
        self.loss = 0

    def forward(self, input):
        self.loss = linear_time_mmd(input, self.target, self.alpha)
        return input


class StyleLossGramMatrix(nn.Module):
    """
    Style Loss Gram matrix
    """
    def __init__(self, target_feature):
        super(StyleLossGramMatrix, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0

    def forward(self, input):
        gram = gram_matrix(input)
        self.loss = F.mse_loss(gram, self.target)
        return input


class StyleLossMeanStdSkewKurtosis(nn.Module):
    """
    Style Loss (Moment Loss) - Mean, Std, Skew, Kurt
    """
    def __init__(self, target):
        super(StyleLossMeanStdSkewKurtosis, self).__init__()
        self.target_mean = compute_i_th_moment(target, 1)
        self.target_std = compute_i_th_moment(target, 2)
        self.target_skewness = compute_i_th_moment(target, 3)
        self.target_kurtosis = compute_i_th_moment(target, 4)

        self.loss = 0

    def forward(self, input):
        input_mean = compute_i_th_moment(input, 1)
        input_std = compute_i_th_moment(input, 2)
        input_skewness = compute_i_th_moment(input, 3)
        input_kurtosis = compute_i_th_moment(input, 4)

        # use balancing factor
        mean_balancing_factor = torch.min(1 / torch.mean(self.target_mean).to(device), torch.tensor(1.).to(device))
        std_balancing_factor = torch.min(1 / torch.mean(self.target_std).to(device), torch.tensor(1.).to(device))
        skew_balancing_factor = torch.min(1 / torch.mean(self.target_skewness).to(device), torch.tensor(1.).to(device))
        kurtosis_balancing_factor = torch.min(1 / torch.mean(self.target_kurtosis).to(device), torch.tensor(1.).to(device))

        self.loss = mean_balancing_factor * F.mse_loss(input_mean.to(device), self.target_mean.to(device))           \
                  + std_balancing_factor * F.mse_loss(input_std.to(device), self.target_std.to(device))              \
                  + skew_balancing_factor * F.mse_loss(input_skewness.to(device), self.target_skewness.to(device))   \
                  + kurtosis_balancing_factor * F.mse_loss(input_kurtosis.to(device), self.target_kurtosis.to(device))

        return input


class StyleLossMeanStdSkew(nn.Module):
    """
    Style Loss (Moment Loss) - Mean, Std, Skew
    """
    def __init__(self, target):
        super(StyleLossMeanStdSkew, self).__init__()
        self.target_mean = compute_i_th_moment(target, 1)
        self.target_std = compute_i_th_moment(target, 2)
        self.target_skewness = compute_i_th_moment(target, 3)

        self.loss = 0

    def forward(self, input):
        input_mean = compute_i_th_moment(input, 1)
        input_std = compute_i_th_moment(input, 2)
        input_skewness = compute_i_th_moment(input, 3)

        # use balancing factor
        mean_balancing_factor = torch.min(1 / torch.mean(self.target_mean).to(device), torch.tensor(1.).to(device))
        std_balancing_factor = torch.min(1 / torch.mean(self.target_std).to(device), torch.tensor(1.).to(device))
        skew_balancing_factor = torch.min(1 / torch.mean(self.target_skewness).to(device), torch.tensor(1.).to(device))

        self.loss = mean_balancing_factor * F.mse_loss(input_mean.to(device), self.target_mean.to(device)) \
                  + std_balancing_factor * F.mse_loss(input_std.to(device), self.target_std.to(device)) \
                  + skew_balancing_factor * F.mse_loss(input_skewness.to(device), self.target_skewness.to(device))

        return input


class StyleLossMeanStd(nn.Module):
    """
    Style Loss (Moment Loss) - Mean, Std
    """
    def __init__(self, target):
        super(StyleLossMeanStd, self).__init__()
        self.target_mean = compute_i_th_moment(target, 1)
        self.target_std = compute_i_th_moment(target, 2)

        self.loss = 0

    def forward(self, input):
        input_mean = compute_i_th_moment(input, 1)
        input_std = compute_i_th_moment(input, 2)

        # use balancing factor
        mean_balancing_factor = torch.min(1 / torch.mean(self.target_mean).to(device), torch.tensor(1.).to(device))
        std_balancing_factor = torch.min(1 / torch.mean(self.target_std).to(device), torch.tensor(1.).to(device))

        self.loss = mean_balancing_factor * F.mse_loss(input_mean.to(device), self.target_mean.to(device)) \
                  + std_balancing_factor * F.mse_loss(input_std.to(device), self.target_std.to(device))

        return input


class StyleLossMean(nn.Module):
    """
    Style Loss (Moment Loss) - Mean
    """
    def __init__(self, target):
        super(StyleLossMean, self).__init__()
        self.target_mean = compute_i_th_moment(target, 1)

        self.loss = 0

    def forward(self, input):
        input_mean = compute_i_th_moment(input, 1)

        # use balancing factor
        mean_balancing_factor = torch.min(1 / torch.mean(self.target_mean).to(device), torch.tensor(1.).to(device))

        self.loss = mean_balancing_factor * F.mse_loss(input_mean.to(device), self.target_mean.to(device))

        return input