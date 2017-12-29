# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission included files to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model. In order to train a new model - it needs to run as -
    python model.py driving_log_directory model_name
    
    Here driving_log_directory is the directory where driving_log.csv file is saved. 
    model_name is the name of model you want to use to train the model. Its value should be nvidia. We also have support of other models like - lenet but it has not generated sufficient score, hence has been removed from model.py to avoid the confusion but BehaviorCloning.ipny has lenet model and others etc.
    
    Run following command to execute the model
    
    ##### python <driving_log_director> nvidia
    
* drive.py for driving the car in autonomous mode - No change has been made in this file.
* model.h5 containing a trained convolution neural network - Final model which has been generated on Nvidia model architecture after template.
* BehaviorCloning.ipynb - ipython notebook having different models (lenet, nvida etc.).

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Here is the model architecture - 
Model architecture consists of - Image transformation (Image cropping) as first layer. cropping2d_1 layer is image cropping layer. 5 Covolution layers follows after cropping layer. Finally Dense layer is added along with drop out layer in middle to avoid overfit. 
This architecture is mainly taken from -https://arxiv.org/pdf/1604.07316.pdf
                ....................................................................................

                Layer (type)                 Output Shape              Param   
                .....................................................................................

                lambda_1 (Lambda)            (None, 160, 320, 3)       0         
                ......................................................................................

                cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
                ......................................................................................

                conv2d_1 (Conv2D)            (None, 86, 316, 24)       1824      
                .....................................................................................

                max_pooling2d_1 (MaxPooling2 (None, 43, 158, 24)       0         
                ....................................................................................

                conv2d_2 (Conv2D)            (None, 39, 154, 36)       21636     
                .....................................................................................

                max_pooling2d_2 (MaxPooling2 (None, 19, 77, 36)        0         
                ......................................................................................

                conv2d_3 (Conv2D)            (None, 15, 73, 48)        43248     
                ......................................................................................

                max_pooling2d_3 (MaxPooling2 (None, 7, 36, 48)         0         
                .....................................................................................

                conv2d_4 (Conv2D)            (None, 5, 34, 64)         27712     
                ......................................................................................

                conv2d_5 (Conv2D)            (None, 3, 32, 64)         36928     
                .....................................................................................

                flatten_1 (Flatten)          (None, 6144)              0         
                .....................................................................................

                dense_1 (Dense)              (None, 100)               614500    
                .....................................................................................

                dropout_1 (Dropout)          (None, 100)               0         
                ......................................................................................

                dense_2 (Dense)              (None, 50)                5050      
                ......................................................................................

                dense_3 (Dense)              (None, 10)                510       
                ......................................................................................

                dense_4 (Dense)              (None, 1)                 11        
                ......................................................................................
##### Total params: 751,419
##### Trainable params: 751,419
##### Non-trainable params: 0
_________________________________________________________________
function  Nvidia_model in model.py describes the model architecture.

#### 2. Attempts to reduce overfitting in the model

In order to resolving the overfitting of model, Dropping layer has been added beween Dense layer 1 and Dense layer 2. model.add(Dropout(0.2)) lines of code in Nvidia_model function represents the dropping layer componenet.

The model has been trained and validated on diffeent sets of data and founds that it does not overfit. Even on trainng time, It is found that training accuracy and validation accuracy of model was minimal even though our validation content of the model was only 10%. 

This model always stay on line while running on simulator. Here is the video link of the simulator captured through Xbox app. https://youtu.be/ZGv59ir5J84

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Training data creation - 

Tranining data was generated with following startegy - 
a. Keeping the vehicle always on track with full speed.
b. Focusing more on curve parts espcially in areas where bounday is open and it seems to be like road.
c. Try to add sufficient data to have representation of all behaviors. For example - we have only one pool in one loop lane. If we drive the pool with maximum speed, Nearly 20-30 images will be generated. While choosing training data, it might be possible that pool behavior might not be available in training data. This may lead to unpredictable behavior. 
I tried to make sure, all behavior is properly represented in training data.
d. Two Lane training data is generated with low speeds and also where cars are moving towards boundary but recovery. This has been done to learn the model to recover in bad situations.

#### 5. Traning data augmentation strategy -

First lane is almost curve towards lane. Learning the model using central image leads to bias learning towards left steering, which is hard for small fraction of lane where it has right turn. Second this is also not good for real world driving as it seems to be overfitted towards left curve.

In order to generate new training data, where we can track of right lane, we negated/flipped the image and augmented the data with right lane curve. This helps to generlize the model.

Camera has mulitple cameras and in order to recover conditions where car is near about boundary, left camera or right camera image is used to agument the training data using correct factor. 

In order to avoid overfit of model, I used only randomly either left or right camera image.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
