# Traffic Sign Classification

[//]: # (Image References)

[demo_new_images]: ./images/demo_new_images.png "Demo New Images"
[train_valid_acc_epoch]: ./images/train_valid_acc_epoch.png "Train and Validation Accuracy Curves"


## Overview


In this project, we will see how deep neural networks and convolutional neural networks are very very powerful to perform image classification tasks. Indeed, we train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, we will then try out your model on images of German traffic signs that we downloaded from the web.

![alt text][demo_new_images] 

You will find the code - using Tensorflow - for this project is in the [IPython Notebook](https://github.com/itismouad/traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb). More details are available by reading the project [notes](https://github.com/itismouad/traffic_sign_classifier/blob/master/traffic_sign_classifier.md).

 

### The final model architecture :


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale/ Normalized image 	   		| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU activation		|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6  	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU activation       |                                               |
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16  	|
| Fully connected		| 400 input, 120 output     					|
| RELU activation       |                                               |
| Dropout               | 0.9 keep probablility (training)              |
| Fully connected		| 120 input, 84 output     				     	|
| RELU activation       |                                               |
| Dropout               | 0.9 keep probablility (training)              |
| Fully connected		| 84 input, 43 output     				     	|



### The final model metrics :


- Training Set Accuracy: **0.9764310717582703**
- Validation Set Accuracy: **0.9523809552192688**
- Test Set Accuracy: **0.93087885996130659**

![alt text][train_valid_acc_epoch]

