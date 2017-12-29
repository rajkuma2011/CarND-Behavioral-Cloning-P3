# **Behavioral Cloning** 


**Behavioral Cloning Project**

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Video link of lane1 - https://youtu.be/ZGv59ir5J84
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
Following parameters are tuned - 
a. optmizer - On try different optimizer, adam optimizer is found best. 
b. learning rate - 0.001 learning is found to be best and converges without flactuating.
c. epochs - 10 ephoch is found to be sufficient.
d. activation function - tanh function converges fast and better than relu activation function.
e. model architecture - Lenet, Nvida and their different variation by adding mulitple dense and convoluation layer is tried. Basic Nvida layer with representive training data, generated the correct model.

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

### 6. Challenges - 
a. While training the model, image is read into BFR format by default due to cv2 library, but drive.py feeds the image in RGB format. This is obvious fix but it costs more than a week to trying different strategy to fix the model.
b. Collecting the repsentative data - For example - sufficient data for lane where we dont have boundary and only available in less than 1% part of one lane. This is handled by adding more training data for correponding region.
c. Figuring out, why it does not work. I found very challenging to dump the output of middle layer in Keras, hence could not figure out, why model is not working.
d. Deciding the archicture of the model. Brute force strategy worked but does not seems to be good way.
7. Model hardly predict 0.0 steering angle but it was predicting near to 0.0. Due to this, car fluctuates alot. We fixed it by normalization the score betwenn -0.01 to 0.01 angle to 0.0. This brough smoothness in the model.

### 7. Improvements - 
a. Model should not fluctuate and should speed up to max speed.
b. Alex net or Google net should be tried using partially trained layer. 
