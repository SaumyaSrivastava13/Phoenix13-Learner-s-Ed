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

**Category - 01   (Academic)**

![Banner-3](https://cdn.discordapp.com/attachments/1046493587916988417/1113105198349553746/3.png)

01. An advanced ranking system that will be updated on a day-to-day basis which would be based on the overall performance during lectures, projects, assignments, and quizzes.

02. Keeping track of the attendance of the student in every subject and sending reminders to attend lectures on subjects in which his attendance is going down at an alarming rate.

03. Providing him/her proper insights about the areas he is currently lacking in along with auto-generated suggestions of resources to study from. 

04. A separate area for assignment submission along with a plagiarism checker. The plagiarism checker would keep on updating itself with each assignment submission. If it receives an assignment with exactly the same content, the student would receive a notification to resubmit another assignment as the presently submitted assignment might be up for plagiarism.

05.  A highlight section that would keep them connected to the world. Giving them a daily insight into what's happening around the globe, especially their areas of interest.

**Category - 02  (Gamification)**

![Banner-4](https://cdn.discordapp.com/attachments/1046493587916988417/1113105577070047252/4.png)

01. Improvisation of the ranking system through day-to-day digital badge system. Every day the student who has outperformed everyone else by giving more quizzes and doing his assignments would be provided with the student of the day batch.

02. Interactive quizzes through facial gestures. Multiple choice  questions would be answered with the movement of the head. 

03. Adding some fun virtual elements, there will be a performance-wise token credit system in this Gamification Centre. Based on this credit, the students can unlock new features for their virtual pet. 

**Category - 03  (Student Insights)**

![Banner-5](https://cdn.discordapp.com/attachments/1046493587916988417/1113105861733253210/5.png)

01. Checking on the mental health of the student, there will be interactive mental health chatbots. In case the student needs any kind of help, he/she can be provided immediate assistance.

02. Detection of drowsy and low behaviour during online lectures. If the student has been feeling drowsy for a certain period of time during the lecture, then some easy sitting exercises will be suggested in order to tackle the drowsiness. 

03. The students can create chatrooms where all the interested can pitch in for a project, assignment, etc. In case they need a mentor, then we can provide them with suggestions from the faculty experts in that field.

## Our Models

![UML-Diagram](https://cdn.discordapp.com/attachments/1046493587916988417/1113139600605585558/Diagram.png)

// Discussing Models with Benchmarking
  
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


