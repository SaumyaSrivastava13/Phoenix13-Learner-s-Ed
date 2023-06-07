# intel-oneAPI

#### Team Name - **Phoenix 13**
#### Problem Statement - **Open Innovation in Education**
#### Team Leader Email - **saumya.srivastava_cs.aiml21@gla.ac.in**

## A Brief of the Prototype:

![Banner-1](https://cdn.discordapp.com/attachments/1046493587916988417/1113101680968466452/1.png)

Quality education is the path to transcendence and making this journey more engaging is the sole purpose of our project. We intend to include features that would provide an interactive interface to the user where he/she can have a more precise watch on his/her performance, explore their areas of interest, and connect with people of the desired field in the vicinity using the real-time trained DL and ML model. 

Making the learning more innovative we would allow them to create chatrooms for discussions on projects, academics, areas of interest, etc., and also help them with auto-generated suggestions. 

Creating fun and amazing pet avatars in which the capital to unlock new features would be through their scores that would be auto-generated based on their day-to-day progress in activities as well as academics.

Apart from learning, we would also take care of their mental health by creating interactive chatbots which could judge a child's mental state and provide suitable help. Thus, providing a pool of exploration alongside fun and healthy competition would help in the overall development of the student.

## How it works?

![Banner-2](https://cdn.discordapp.com/attachments/1046493587916988417/1113103428026118226/2.png)

The user interactions related to learning would be duly recorded. These recordings will be taken as input features for our Intel oneAPI oneDNN and oneDAL integrated ML and DL models. These models will further make predictions, cluster the students according to their areas of interest, generate visualization for their progress in each of their domains/subjects, and provide the candidates with study material recommendations whether it is a video, blog, or book.
Some fun activities and assessments to help them, co-op with their studies, which in turn will be recorded and a ranking system amongst the students say class-wise, section-wise, or campus-wise will provide them scores for performing well in those activities and classes and rank them accordingly, creating a healthy competitive environment.

During Covid-19, when everything had shifted to online mode, students used to get bored and drowsy during classes and it was not possible to keep a check on each and every student. Through our project, we would train the model to check if the person is getting drowsy and recommend him some auto-generated exercises that would help him get his focus back. Most institutions stick to the old school analysis systems in which the student is never able to understand specifically in which area, he/she should improve in order to maintain his academic results, but through our project, he/she can get a detailed analysis of his performance with day to day updates which will keep him/her more aware about his situation in every area.

# List of features offered by the solution

## Vision

Making learning more interactive and innovative by providing a clearer picture along with some fun elements, here we list the features of our project bifurcated into three categories.

## Category - 01   (Academic)

![Banner-3](https://cdn.discordapp.com/attachments/1046493587916988417/1113105198349553746/3.png)

01. An advanced ranking system that will be updated on a day-to-day basis which would be based on the overall performance during lectures, projects, assignments, and quizzes.

02. Keeping track of the attendance of the student in every subject and sending reminders to attend lectures on subjects in which his attendance is going down at an alarming rate.

03. Providing him/her proper insights about the areas he is currently lacking in along with auto-generated suggestions of resources to study from. 

04. A separate area for assignment submission along with a plagiarism checker. The plagiarism checker would keep on updating itself with each assignment submission. If it receives an assignment with exactly the same content, the student would receive a notification to resubmit another assignment as the presently submitted assignment might be up for plagiarism.

05.  A highlight section that would keep them connected to the world. Giving them a daily insight into what's happening around the globe, especially their areas of interest.

## Category - 02  (Gamification)He is not.

![Banner-4](https://cdn.discordapp.com/attachments/1046493587916988417/1113105577070047252/4.png)

01. Improvisation of the ranking system through day-to-day digital badge system. Every day the student who has outperformed everyone else by giving more quizzes and doing his assignments would be provided with the student of the day batch.

02. Interactive quizzes through facial gestures. Multiple choice  questions would be answered with the movement of the head. 

03. Adding some fun virtual elements, there will be a performance-wise token credit system in this Gamification Centre. Based on this credit, the students can unlock new features for their virtual pet. 

## Category - 03  (Student Insights)

![Banner-5](https://cdn.discordapp.com/attachments/1046493587916988417/1113105861733253210/5.png)

01. Checking on the mental health of the student, there will be interactive mental health chatbots. In case the student needs any kind of help, he/she can be provided immediate assistance.

02. Detection of drowsy and low behaviour during online lectures. If the student has been feeling drowsy for a certain period of time during the lecture, then some easy sitting exercises will be suggested in order to tackle the drowsiness. 

03. The students can create chatrooms where all the interested can pitch in for a project, assignment, etc. In case they need a mentor, then we can provide them with suggestions from the faculty experts in that field.

## Our Models

![UML-Diagram](https://cdn.discordapp.com/attachments/1046493587916988417/1113141041399332874/Diagram.png)

# Student Login/Register

Here we have the student login and registration page. Here, the newcomer student need to register themselves and enjoy the benefits of the dashboard. The new registration student needs to give the favorite categories, which will be used later on by our recommendation system to fetch the recommended videos for them.
Also, here is the login page for the student who are already registered. They need to fill their email and password, and then enjoy the benefits of the portal services.

#### Registration Page

![Register](https://cdn.discordapp.com/attachments/1046493587916988417/1115982762248261672/Screenshot_2023-06-07_180649.png)

#### Login Page
![Login](https://cdn.discordapp.com/attachments/1046493587916988417/1115982762701226004/Screenshot_2023-06-07_180632.png)

# Dashboard

Now the new student is welcomed with the dashboard welcome message. Here we have the weather API that would Fetch the weather and keep the dashboard more updated with the outside world, so that student feel interacted with the world as well. Now, along with this, there is the rank of the student and also three graphs of the ranks amongst class wise, section wise and school wise. So that students can always keep track of where they stand out in the competing environment.
Now there will be a alert or notification section as well, where the alerts from the admins will be presented. The next section includes the attendance for all the subjects here the average attendance is also mentioned. Now going to the next section, here is the Assignment Updation Center, where the student will get to know how many assignments are still left to do. After that, we have the LearnersEd Coin Bank, which tells us how many coins are left with the user And on the right side we have our virtual pet information, such as pet name, level and rank amongst these students and also a graph which gives the estimation of rank amongst others.
At last, we have the recommendation section where the category selected would determine which type of videos will be fetched for the student and be presented in the recommendation section. 

![Dash-1](https://cdn.discordapp.com/attachments/1046493587916988417/1115985017231917157/Screenshot_2023-06-07_181401.png)

![Dash-2](https://cdn.discordapp.com/attachments/1046493587916988417/1115985016892170423/Screenshot_2023-06-07_181456.png)

![Dash-3](https://cdn.discordapp.com/attachments/1046493587916988417/1115985016548245504/Screenshot_2023-06-07_181545.png)

# Lecture Section

Now we have the lecture section where all of the lectures recommended for the students are listed here. The student can watch that lecture by clicking on the watch button after going into the lecture section. Our drowsiness detection system would automatically work and will detect if the student is drowsy or awake. If the student is drowsy, then it would recommend some random sitting exercises from the database. The attendance will not be granted if the students is continuously being predicted drowsy. This makes the job of teachers more easier, as the students by themselves will be provoked to be aware of the lectures.

#### Lecture List

![Lect](https://cdn.discordapp.com/attachments/1046493587916988417/1115989285913493525/Screenshot_2023-06-07_183042.png)

#### Not Drowsy

![lec-1](https://cdn.discordapp.com/attachments/1046493587916988417/1115989285108187167/Screenshot_2023-06-07_183229.png)

#### Drowsy

![lec-2](https://cdn.discordapp.com/attachments/1046493587916988417/1115989285443735703/Screenshot_2023-06-07_183128.png)

# Drowsiness Detection System
 
You can acces the drowsiness detection model used behind this, present in the AIML Modules Folder.

The provided code in the AIML modules folder implements a drowsiness detection model using Convolutional Neural Networks (CNNs). Here is a summary of the code functionality:

Importing necessary libraries: The code starts by importing the required libraries, including PIL, OpenCV, face_recognition, TensorFlow, and Keras.

Eye Cropping Function: The eye_cropper function takes an image path as input and uses OpenCV and face_recognition to locate the eyes in the image. It crops the eye region, resizes it to 80x80 pixels, and returns the cropped image for further processing.

Loading Images from Dataset: The load_images_from_folder function loads images from the specified folder and resizes them to 80x80 pixels. It assigns a label (0 for open eyes, 1 for closed eyes) and creates a list of image-label pairs.

Preparing the Dataset: The code creates arrays for input images (X) and corresponding labels (y). It iterates through the image-label pairs, appends the images to X, and labels to y. The images are reshaped, normalized, and the labels are converted to arrays.

Splitting the Dataset: The dataset is split into training and testing sets using the train_test_split function from sklearn. The splitting is stratified based on the labels to maintain class balance in both sets.

Model Definition: The code defines the CNN model using the Sequential API of Keras. It includes convolutional layers, max-pooling layers, dense layers, and dropout layers for regularization.

Model Compilation and Training: The model is compiled with binary cross-entropy loss and Adam optimizer. It is then trained on the training data, using the fit function, for 24 epochs. The validation data is used to monitor the model's performance during training.

Model Evaluation: The trained model is evaluated on the testing data using the evaluate function. The evaluation results, including loss and metrics, are printed.

Model Saving: The trained model is saved to a file using the save function.

Prediction Function: The model_response function takes an image and uses the eye_cropper function to extract the eye region. The preprocessed image is then passed to the trained model for prediction. If the predicted probability of closed eyes exceeds a threshold, the function returns 'Yes,' indicating drowsiness.

Model Usage: The model_response function is called with an image to demonstrate the usage of the trained model.

In summary, the code prepares and trains a CNN model to classify eye states as open or closed for drowsiness detection. It provides a function to extract eye regions from images and a function to classify the eye state using the trained model.

# Intel Optimization Applied

Along with the normal model we have applied Intel oneDNN with OpenMP and scikit-learn-intelex, the scikit learn optimization by intel, which further leverages our model preformance. Using these optimization tools helped us getting more inferences in less time and train our model very fast as well

Note: We have trained our model using intel oneAPI AI analytics toolkit oneDNN and OpenMP on intel i5 11th gen 11260H 6 core 12 thread computer.

Here are the OpenMP params used

inter: 6

intra: 6

KMP_BLOCKTIME: 1

Test_Set: 25

#### Benchmarks Rates

Inference Time Rate: 1.1440191387559806

Latency Rate: 0.8741112505227936

Throughput Rate: 1.1440191387559806

Training Time Rate: 0.438332327047551

Here is the benchamarking difference between the model trained on normal cpu vs on intel optimization, same hardware better performance!

![banch](https://cdn.discordapp.com/attachments/1046493587916988417/1115992633936986182/drowsy.png)


# Gamify Section

Now coming to the most entertaining section of the LearnersEd Portal, the gamify section. Here the students are encouraged to sharpen their minds along with having a fun competitve environment amongst them. There are two section here gamify quiz and virtual pet.

![gam](https://cdn.discordapp.com/attachments/1046493587916988417/1116014066574569564/Screenshot_2023-06-07_185607.png)

## Gamify Quiz 

Here the students are presented with a quiz, but the twist is the answers are not selected by mouse or keyboard input, the options are selected by their head posture and this would reduce their decision taking time and would improve their reflexes to act upon situations, every student needs to attentively solve the quiz and then after the quiz they would get Learners'Ed Coins according to the marks they scored.

![q](https://cdn.discordapp.com/attachments/1046493587916988417/1116014066285170708/Screenshot_2023-06-07_185628.png)

![q2](https://cdn.discordapp.com/attachments/1046493587916988417/1116015387583205478/Screenshot_2023-06-07_201413.png)

![q3](https://cdn.discordapp.com/attachments/1046493587916988417/1116015386983407636/Screenshot_2023-06-07_201617.png)

This module is supported by Face Pose detection model developed by us.

The provided code performs head pose estimation using a machine learning model. Here is a technical write-up summarizing its functionality:

Data Loading: The code loads the input data from a pickle file. It consists of images as input samples (x) and corresponding head pose angles (y).

Data Preprocessing: The head pose angles (y) are split into three components: roll, pitch, and yaw. The code then prints the minimum, maximum, mean, and standard deviation values for each component to provide insights into the data distribution.

Dataset Splitting: The input data (x and y) is split into training, validation, and testing sets using the train_test_split function from scikit-learn. The training set contains 70% of the data, and the remaining 30% is evenly divided between the validation and testing sets.

Data Standardization: The StandardScaler is applied to standardize the input features (x_train, x_val, and x_test) by subtracting the mean and scaling to unit variance.

Model Architecture: The code defines a neural network model using the Sequential API from Keras. The model consists of three dense layers with ReLU activation. The first two layers have regularization using L2 kernel regularization.

Model Compilation and Training: The model is compiled with the Adam optimizer and mean squared error (MSE) loss function. It is trained on the training data with early stopping based on the validation loss. The training progress is stored in the hist variable.

Model Saving: The trained model is saved to a file named "model.h5".

Model Evaluation: The code evaluates the trained model on the training, validation, and testing sets, printing the loss values for each set.

Visualization: The training and validation loss curves are plotted to visualize the model's training progress.

Face Point Detection: The code defines a function detect_face_points that uses the dlib library to detect the 68 facial landmarks on an input image.

Feature Computation: The compute_features function calculates pairwise Euclidean distances between the detected facial landmarks, resulting in a feature vector.

Feature Standardization: The computed features are standardized using the same StandardScaler instance used for the input features.

Model Loading: The saved model is loaded from the "model.h5" file.

Head Pose Estimation: The standardized features are fed into the loaded model to predict the roll, pitch, and yaw angles of the head pose.

Result Visualization: The input image is displayed with the detected facial landmarks and the predicted head pose angles.

In summary, the code performs head pose estimation by training a neural network on a dataset of images and corresponding head pose angles. It preprocesses the data, builds and trains the model, and then uses the trained model to predict head pose angles for new input images. The detected facial landmarks and predicted head pose angles are visualized for analysis and interpretation.

# Intel Optimization Applied

Here we applied oneDNN along with OpenMP and scikit-learn extension which leverages performance for us.

Here are the OpenMP params

inter: 2

intra: 6

KMP_BLOCKTIME: 0

Test_Set: 25

#### Benchmark Rate

Inference Time Rate: 1.0136551020270463

Latency Rate: 0.9865288479289062

Throughput Rate: 1.0136551020270463

Training Time Rate: 0.4256577027908266

Here is the benchmarks difference in normal cpu vs intel optimized cpu

![bgam](https://cdn.discordapp.com/attachments/1046493587916988417/1116017222951903284/FacePose.png)

//

![Process](https://cdn.discordapp.com/attachments/1046493587916988417/1115980341925130251/Intel_oneAPI_Hackathon_PPT.png)
  
## Tech Stack: 

#### List of oneAPI AI Analytics Toolkits & its libraries used

**Intel oneAPI Base Toolkit**
(General Compute)

1) Intel® oneAPI Data Analytics Library
2) Intel® oneAPI Deep Neural Networks Library
3) Intel® Distribution for Python
4) Intel® oneAPI Math Kernel Library

**Intel® AI Analytics Toolkit**
(End-to-End AI and Machine Learning Acceleration)

1) Intel® Distribution for Python with highly optimized scikit-learn
2) Intel® Optimization for TensorFlow
3) Intel® Optimization of Modin

**Base Technology Stack**
1) HTML & CSS - Web Application (Frontend)
2) Tailwind CSS - (Style)
3) Django - Web Application (Backend)
4) Javascript - Validation & Client-Side Scripting
5) MongoDB - DBMS
6) Matplotlib & Seaborn - Data Visualization
7) Google Charts, Charts.js and/or any other 3rd Party - Data Visualizer
8) TensorFlow & scikit-learn (scipy) - DL and ML Model 
9) OpenCV - Computer Vision
   
## Step-by-Step Code Execution Instructions:
// will be done after whole prototype being pushed
  This Section must contain set of instructions required to clone and run the prototype, so that it can be tested and deeply analysed
  
## What I Learned:
This advanced learning platform has motivated us to excel in every aspect of this project and upskill our knowledge parameters in various areas. Some of them are:-

**1) Development Environment Handling & Management:**

With the help of the conda package manager and the default environment creation commands (windows), we learned, how to create and manage these environments in order to accomplish desired tasks. With the help of these environments, we were able to benchmark our model on different packages, with and without oneAPI integrations and optimizations.

**2) Benchmarking & Inference Testing**

Having a knowledge background of machine learning and deep learning we had some information about the performance and throughput differences between various algorithms. However, the major factor that helped us take our models to the maximum level of optimization is the benchmarking technique. In order to perform benchmarking, we visualized the differences between various crucial parameters that acted as the deciding factors with the help of graphs and charts, for choosing a more optimized and faster model.

We also analyzed the bechmarking between model with and without using Intel oneAPI integration and optimization, as one can guess, the model build with integration of Intel oneAPI were able to get our model to the most efficient training and least inference time to predict the output. 

We use different environments, one with oneAPI enabled and other with oneAPI being disabled. The advance optimization libraries of AI Analytics Toolkit, such as Extension for scikit-learn, Extension for Pandas (Modin), Intel Distribution for Python, OpenMP (Open Multi-Processing), oneMKL (Math Kernel Library), oneDNN (Deep Neural Network), oneDAL (Data Analytics Library) and Intel Optimization for Tensorflow Extension, we were able to create a huge difference between the inference time and training time against the model not built with the Intel oneAPI Optimization. 

For benchmarking, we engadged our local system environment in order to obtain the respective paramenters. We also used W&B (Weights & Biasis) wandb library for real-time visualization of hardware paramerers like processor memory usage (cores), and processor threads etc.

**3) Model Web Deployment**

As we decided to develop our prototype through web development, so in order to integrate our ML and DL trained models, we needed a strong backend and an appealing frontend. For backend technologies, we used Django, which helped us in integrating our models with a client-side visual frontend and obtain inferences for given set of inputs.

**4) Advanced Database Management System**

As we are advancing in the current surge of Artificial Intelligence and Machine Learning, data became an important aspect of our lives. To maintain this data in huge storage is also a matter of concern. But, considering the fact, that this data also contains faulty data, which can include extra null values, blank data cells, wrong inputs, etc. To manage these unnecessary data and obtain a more space/memory efficient Database Management System, here in our project, we used MongoDB, which is a Document Oriented Database Management System which helps use to only use the memory space if the input provided is available. This Document Orineted model provides best output for multi-media file datatypes as well, which is a major need in AIML Web Deployment project.



<h1 align="center">Thank You</h1>


