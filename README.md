# Style Visualizations

This project implements the a style visualization algorithm proposed by Mahendran
et al. in [1]. Gradient descent is performed on a noise image to
match the (style) features of a style image produced by an encoder network. 

- The project is written in Python ```3.7``` and uses PyTorch ```1.1``` 
(also working with PyTorch ```1.3```).

- ````requirements.txt```` lists the python packages needed to run the 
project. 

### Network

The network architecture resembles a common encoder architecture. A VGG-19
network is utilized and produces the feature responses of a noise image and 
a style image. 

### Loss
 
A style loss ensures the transfer of style structure from the style image to the 
noise image. Gradient descent is performed on the noise image to match the statistics
given by the style loss. A style loss measures the MSE between the Gram matrix of noise
features and the Gram matrix of style features summed up over the specified layers 
in the network.

#### Moment Alignment
The idea of the project is to compare the Gram matrix loss with a so-called moment 
loss that measures the difference between the first ````N```` moments of noise
features and style features.  

A further explanation as well as analysis can be found in the pdf-file of this 
repository.

### Usage

The ``configurations``-folder specifies three configurations that can be used to 
produce transfer image. The project only gets the exact path to the 
configuration that is used e.g. ```python main.py './configurations/train.yaml'```.

- ``train.yaml`` produces style visualizations from the specified path with the 
moment loss.

- ```train_mmd.yaml``` produces style visualizations where the MMD 
is utilized as loss function. Please also see [2].

- ```train_gram.yaml``` produces style visualizations with the original by Gatys 
et al [3] proposed Gram matrix loss as a baseline. 

### Additional Information

This project is part of a bachelor thesis which was submitted in August 2019. The 
style visualizations network makes up one chapter of the final thesis. A slightly modified 
version of the chapter can be found in this repository as a pdf-file. Also, the chapter 
introduces all related formulas to this work. 

The final thesis can be found [here](https://jzenn.github.io/projects/bsc-thesis) in a corrected and modified version.

### References

[1] A. Mahendran and A. Vedaldi. Understanding deep image representations 
by inverting them. In *The IEEE Conference on Computer Vision and 
Pattern Recognition (CVPR), 2015*.

[2] A. Gretton, K. Borgwardt, M. Rasch, B. Schölkopf, and A. Smola. 
A kernel two-sample test. In *J. Mach. Learn. Res., 2012*.

[3] L. Gatys, A. Ecker, and M. Bethge. Texture synthesis using 
convolutional neural networks. In *Conference on Neural Information 
Processing Systems (NIPS), 2015*.