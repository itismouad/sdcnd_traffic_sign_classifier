## Traffic Sign Recognition


In this project, we will see how deep neural networks and convolutional neural networks are very very powerful to perform image classification tasks. Indeed, we train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, we will then try out your model on images of German traffic signs that we downloaded from the web.

You will find the code - using Tensorflow - for this project is in the [IPython Notebook](https://github.com/itismouad/sdcnd_traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynb). An HTML version of the file is also available for visualizing the results.


[//]: # (Source Images)

[demo_original]: ./images/demo_original.png "Original Demo Images"
[demo_preprocessed]: ./images/demo_preprocessed.png "Preprocessed Demo Images"
[demo_rescaled]: ./images/demo_rescaled.png "Rescaled Demo Images"
[demo_rotated]: ./images/demo_rotated.png "Rotated Demo Images"
[demo_translated]: ./images/demo_translated.png "Translated Demo Images"
[demo_random_augmented]: ./images/demo_random_augmented.png "Ramdomly Augmented Demo Images"
[demo_new_images]: ./images/demo_new_images.png "New Demo Images"
[demo_blurred]: ./images/demo_blurred.png "Blurred Demo Images"
[original_dataset_dist]: ./images/original_dataset_dist.png "Original Dataset Distribution"
[augmented_dataset_dist]: ./images/augmented_dataset_dist.png "Augmented Dataset Distribution"

[class_prob0]: ./images/class_prob0.png "Class Probabilities 1"
[class_prob1]: ./images/class_prob1.png "Class Probabilities 2"
[class_prob2]: ./images/class_prob2.png "Class Probabilities 3"
[class_prob3]: ./images/class_prob3.png "Class Probabilities 4"
[class_prob4]: ./images/class_prob4.png "Class Probabilities 5"
[class_prob5]: ./images/class_prob5.png "Class Probabilities 6"

[lenet-5]: ./images/lenet-5.png "lenet-5 architecture"

[train_valid_acc_epoch]: ./images/train_valid_acc_epoch.png "Train and Validation Accuracies"


### Data Set Summary & Exploration

Let's first explore at the dataset that we will use to perform the traffic sign classification task. Below are listed the basic dataset summary statistics :

- Number of training examples = `34799`
- Number of validation examples = `4410`
- Number of testing examples = `12630`
- Image data shape = `(32, 32, 3)`
- Number of classes = `43`


We will keep the same demo images throughout these notes for simplicity and to make sure the reader understands the transformations performed during the dataset pre-processing/augmentation.

![alt text][demo_original]

The quality of the pictures are very variable : pictures are not all centered, they do not have the same brightness, they do not have the same quality, etc.

Moreover, if we look at the distribution of the classes, we realize that our dataset is extremly imbalanced. Some of the classes have nearly seven times as much training examples as the least  represented class. We will fix this issue later by performing dataset augmentation but let's focus on the pre-processing steps for now.

![alt text][original_dataset_dist]


### Data Pre-processing and Dataset Augmentation

It is usual to pre-process the data to make sure the model sees "normalized" data. For images, it is common to perform grayscaling (reduce the number of channels - from 3 to 1), normalization by mean substraction, etc.

**Grayscaling and Normalization**

To implement grayscaling of the traffic sign images, we use the very useful `cv2.cvtColor` function from the opencv library. We can indeed assume that the colors do not contain the main information about the traffic sign identity. We would rather have our artifical neural network focus on the geomtery of the sign to extrat the correct label. 

A common preprocessing techniques in machine learning is also normalization. We implement here too.

Images after grayscaling and normalization:
![alt text][demo_preprocessed]


**Dataset Augmentation**

Another issue we detected with our dataset is how imbalanced it is. To soove this problem, I decided here to  I decided here to augment the dataset by implementing several image transformations that will allow to enhance our dataset :

- translation
- rescaling
- rotation
- blurring

In practice, if a class has less pictures than the dataset class average, we add "new" pictures until reaching this number. We add those pictures randomly. You will find below the effect of those transformation on our demo images : 

Images after translation transformation :
![alt text][demo_translated]

Images after rotation transformation :
![alt text][demo_rotated]

Images after blurring transformation :
![alt text][demo_blurred]

Images after rescaling transformation :
![alt text][demo_rescaled]


At this point, the distribution of our classes is completely modified. The number of training examples increases from `34799` to `46714`. See below the new distribution :

![alt text][augmented_dataset_dist]

Let's now work on the design and architecture of our model.

### Design and Test a Model Architecture

I inspired myself from the [LeNet-5 architecture](http://yann.lecun.com/exdb/lenet/) that was first published by Yann Lecun's lab in 1998.

![alt text][lenet-5]

I edited it by adding dropouts layers for instance. This improves drastically over-fitting, issue that I faced very early in the training process. Other than that, I tried training the model with 3 channels but this did not lead at all to better accuracy. I also played with the hyperparameters, but in the end, augmenting the imbalanced dataset and the regularization were the most effectives changes done to the LeNet-5 architecture.

Specifically, the model had the following setup:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale and Normalized image 	   	| 
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

My model architecture using the hypperparamters settings `EPOCHS = 25`, `BATCH_SIZE = 128`, and `LEARNING_RATE = 0.001` allowed me to reach the following accuracies :

- Training Set Accuracy: **0.9764310717582703**
- Validation Set Accuracy: **0.9523809552192688**
- Test Set Accuracy: **0.93087885996130659**

![alt text][train_valid_acc_epoch]

All the tensorflow details are available in the python notebook. For instance, in terms of the optimizer used, I made sure to use the Adam optimizer which is my "go-to" optimizer which performs usuaaly much better than Stochatic Gradient Descent, AdaGrad, etc. ([see more](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)). 

### Test a Model on New Images

After all those efforts, we can now check the performance of our model on random pictures of German traffic signs downloaded from the Internet.

![alt text][demo_new_images]

I plotted the top 5 predictions (based on softmax probabilities) for the 6 different images. I used the very handful function [tf.nn.top_k](https://www.tensorflow.org/api_docs/python/tf/nn/top_k) that finds values and indices of the k largest entries for the last dimension. Applied to the logits layer, we get the top class probablilities.

![alt text][class_prob0]
![alt text][class_prob1]
![alt text][class_prob2]
![alt text][class_prob3]
![alt text][class_prob4]
![alt text][class_prob5]

From the charts above, we can see that the model is very good at identifying any traffic sign. The probabilities are very high (greater than 90% in most of the cases).

| Image			              | Prediction	        						| 
|:---------------------------:|:-------------------------------------------:| 
| Right-way of intersection   | Right-way of intersection   				| 
| Speed limit (30km/h)        | Speed limit (30km/h)   						| 
| Priority Road  			  | Priority Road   							| 
| Stop sign 				  | Stop sign  									| 
| Road work				      | Road work   								| 
| Turn right ahead  		  | Right-way of intersection   				| 

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of 83.3%


### Final thoughts


Deep convolutional neural networks are mind-blowing when it comes to visual recognition. This project allowed me to verify it in practice using tensorflow and GPU instance on AWS.

I plan in the next steps to : 

- further tune the hyperparameters
- visualizing the Neural Network layers outputs to understand how the model classifies in practice the images
- explore more achitectures

Thank you for reading. ✌️