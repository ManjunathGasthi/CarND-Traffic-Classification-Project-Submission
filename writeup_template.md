# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[sign1]: ./examples_training/sign1
[sign2]: ./examples_training/sign2
[sign3]: ./examples_training/sign3
[sign4]: ./examples_training/sign4
[sign5]: ./examples_training/sign5
[sign6]: ./examples_training/sign6
[sign7]: ./examples_training/sign7
[sign8]: ./examples_training/sign8
[sign9]: ./examples_training/sign9
[sign10]: ./examples_training/sign10
[sign11]: ./examples_training/sign11
[sign12]: ./examples_training/sign12

[sign_histogram]: ./examples_training/signs_hist.png

[sign1_g]: ./examples_training_gray/sign1_g
[sign2_g]: ./examples_training_gray/sign2_g
[sign3_g]: ./examples_training_gray/sign3_g
[sign4_g]: ./examples_training_gray/sign4_g
[sign5_g]: ./examples_training_gray/sign5_g
[sign6_g]: ./examples_training_gray/sign6_g
[sign7_g]: ./examples_training_gray/sign7_g
[sign8_g]: ./examples_training_gray/sign8_g
[sign9_g]: ./examples_training_gray/sign9_g
[sign10_g]: ./examples_training_gray/sign10_g
[sign11_g]: ./examples_training_gray/sign11_g
[sign12_g]: ./examples_training_gray/sign12_g

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used NumPy to calculate summary statistics of the traffic signs data set, as well as some basic Python functions:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I looked at two aspects; distribution of color in sample images; looking for any patterns as shown in the following file

[sign_histogram]: ./examples_training/signs_hist.png

There was nothing special there.

Then I looked at the number of images per category; as shown in the attached files

[categories_dots]: ./examples_training/categories_dots.png
[categories_bars]: ./examples_training/categories_bars.png

It became clear that there is a large gap in each category presentation.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I tried to run the images as they are (3 channels) on the LeNet models; but got very weak results. So I normalized the images; as it is mentioned in the lecture that normalization helps improve the solution convergence process.

I retried the model; still with 3 channels; and added 1x1 convlutions to reduce the channels to fit the starting point of 32x32x1 for the model, but got results in the range 5x%. So I decided to gray the images; then the numbers jumped to the hight 80s.

Here are some examples of traffic sign images before and after grayscaling.
[sign1]: ./examples_training/sign1
[sign2]: ./examples_training/sign2
[sign3]: ./examples_training/sign3

[sign1_g]: ./examples_training_gray/sign1_g
[sign2_g]: ./examples_training_gray/sign2_g
[sign3_g]: ./examples_training_gray/sign3_g

I decided to get additional data to balance the data representation in the input; so the results improved. When I added the extra cases; the validation results moved atop 95% across the board. I got these cases from the training set provided on the GTSRB. 

However; testing external cases still performed very poor; 0% success; so I augmented the data by shearing, translating and 
rotating the images. Again increasing the number of under represented categories during the process.

This has improved both the original testing results as well as the external test cases.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 24x24x16  	|
| RELU					|												|
| Max Pool K 2x2   		| 2x2 stride, valid padding, outputs 12x12x16	|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x12  	|
| RELU					|												|
| Fully connected		| Input: 1200, Output: 200        				|
| RELU					|												|
| Fully connected		| Input: 200, Output: 120        				|
| RELU					|												|
| Fully connected		| Input: 120, Output: 43        				|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a learning rate of 0.001, batch size of 128, 100 Epochs, an Entropy loss function, and an Adam optimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.4%
* validation set accuracy of 98.3% 
* test set accuracy of 94%

I started with straightforwad LeNet; only changing the final output to 43 instead of 10; this gave validation accuracy of 89% before any preprocessing; after grayscaling and normalization the results improved to 97%. However, the model performed very bad on the external data 00% success. 

I tried different models as explained below; they all produced better the required 93% during validation. 

The results were fine, but I tried different other alternatives as follows:
* LeNet: (As mentioned above) Original architecturel; with change of final output size only.
* LeNet_1: I short-cicuit the first convolution layer to the first fully connected layer.
* LeNet_2: I removed the 1st pooling layer.
* LeNet_3: Similar to the original LeNet but adding a direct input from the 1st convolution to the final classifier.
* LeNet_4: I added a dropout layer after I noticed that the performance with the new test images from the web was bad.

Only the original model and model 2 produced more than 90% in testing, so I decided to continue with them.

However; the results on external test cases remained at 0%, so I augmented the original data; as explained above. This produced 35% for the original model, and 57% for model 2.

So model_2 is the best across the board; indicating that the original LeNet was underfitting. So, it is the best model for me.

I am including the code for all models and their behavior in validation and testing; as well the behavior of 2 models for the external test cases - Original and model 2.

If a well known architecture was chosen:
* I used LeNet as a base model.
* I thought it would be relevant to this problem as per the following paper [http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf]
* The final accuracy of the model scored 98.7% on validation; reaching 99% in some of the epochs during training; it also reached 9x% on testing; whcih indicated that it wasn't overfitting in this context. Still on external data; it underperformed significantly. Which I believe is more relevant to the original data set preparation; as I only resized the external data that I chose.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 14 German traffic signs that I found on the web, with comments on there quality where applicable:

![speedlimit30.jpeg][External_photos/01/speedlimit30.jpeg]
![speedlimit70.jpeg][External_photos/04/speedlimit70.jpeg]
The number "70" is too distorted; espcially the "7"

![nopassing.jpg][External_photos/09/nopassing.jpg]
![priorityroad.jpeg][External_photos/12/priorityroad.jpeg]
![yield.jpeg][External_photos/13/yield.jpeg]
![stop.jpeg][External_photos/14/stop.jpeg]
![novehicles.jpeg][External_photos/15/novehicles.jpeg]
![noentry.jpeg][External_photos/17/noentry.jpeg]
![doublecurve.jpeg][External_photos/21/doublecurve.jpeg]
![roadwork.jpeg][External_photos/25/roadwork.jpeg]
![trafficsignals.jpeg][External_photos/26/trafficsignals.jpeg]
The sign is deeply hidden in the background.

![childrencrossing.jpeg][External_photos/28/childrencrossing.jpeg]
![aheadonly.jpeg][External_photos/35/ahedonly.jpeg]
The sign is too close to the ground.

![roundaboutmanadatory.jpeg][External_photos/40/roundaboutmandatory.jpeg]
The lighting is reversed; the image is too dark compared to the background.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Vehicles      		| No Vehicles   								| 
| Children Crossing     | Children Crossing 							|
| Ahead Only 			| Keep Left										|
| Traffic Signals	    | Double Curve					 				|
| Speed Limit 30 Km/ h	| Speed Limit 30 Km/h      						|
| Road Work 			| Road Work 									|
| Yield     			| Yield											|
| Speed Limit 70 Km/h 	| Double Curve									|
| Stop      			| Stop											|
| Priority Road 		| No Passing									|
| Roundabout Mandatory 	| Priority Road									|
| No Entry  			| No Entry										|
| Double Curve 			| Double Curve									|
| No Passing 			| Ahead Only									|


The model was able to correctly guess 8 of the 14 traffic signs, which gives an accuracy of 57%. This compares infavorably to the accuracy on the test set of 94%.

In general; the quality of these photos is different from that of the GTSRB site; it seems that some other preprocessing was applied to the images; while I only resized the downloaded photos before going through my pipeline.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is very sure that this is a 'no vehicles' sign (probability of ~ 1), and the image does contain a 'no vehicles' sign. The rest of the probablities are ~ 0.

The model in general is very certain of its predictions; for all the images; except the last one; which I consider a weakness given its low accuracy. 

In the notebook; I am also showing the performance of the original LeNet scenario; which has lower certainties; but also lower prediction accuracy.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


