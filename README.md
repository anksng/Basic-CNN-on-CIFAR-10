# Basic-CNN-on-CIFAR-10
Building a CNN in TensorFlow to classify images in CIFAR-10 dataset and comparing models with different configurations to see the effect on accuracy.

# Goal 
* To learn handling of image data and pre processing
* See what *Convolutional layers* actually outputs
* See the effect of *maxpool* and *batch normalization* layer
* Train CNN's with different configurations and plot train/test accuracies.

# Layers configurations used 
### Config 1:
![alt text](config1.png)
#### Shapes: 
To build a CNN with tensorflow we must understand the shapes of array at each layer as they must be accurately specified. Also this helps understand the effects of *padding* and *strides* in convolution layer. Following are the shapes of the image as it passes through the config1 layers : 
* Input size = 32 * 32 * 3   (size of image in CIFAR-10)
* Flatten = COnverts the shape 32 * 32 * 3 to  [ *Number of images*, 3072 ]
### Config 2:
![alt text](config2.png)



