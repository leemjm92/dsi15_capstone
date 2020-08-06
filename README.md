# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project : Steering Angle Prediction

##### by: Lee Melvin, SG-DSI-15

## Problem Statement

To develop a Convolutional Neural Network-based regression model, based on Highway Driving Footage to determine if the model is capable of predicting the steering wheel angle. The evaluation metrics RMSE (root mean square error) where we try to minimise the value as much as possible.

## Executive Summary

The push for self-driving vehicles has increased dramatically over the last few years after the success of the DARPA
Grand Challenge 2. An autonomous vehicle works seamlessly when multiple components come together and work in synergy. The most important parts are the various sensors and the AI software powering the vehicle. One of the most critical functions of the AI is to predict the steering angle of the vehicle for the stretch of road lying immediately in front of it and accordingly even steer the car. Increase in computational capabilities over the years allows us to train deep neural networks to become the ”brain” of these cars which then understands the surroundings of the car and makes navigation decisions in real time.   

Comma.ai released 7 hours of driving footage which is available for open source usage. The data set of images was taken while a car was being manually driven, annotated with the corresponding steering angle applied by the human driver. The goal of one of the challenges was to find a model that, given an image taken while driving, will minimize for RMSE (root mean square error) between the model predictions and the actual steering angle.   

In this project, we explore deep convolutional neural networks to predict steering angle values. Predicting the steering angle is one of the most important parts of creating a self-driving car. This task would allow us to explore the full power of neural networks, using only steering angle as the training signal, deep neural networks can automatically extract features from the images to help understand the position of the car with respect to the road to make the prediction.  

Based on the dataset from comma.ai, my model was able to attain an overall RSME score of 0.805. 

 ## Conclusions and Recommendations

The steering prediction problem is form of regression problems. It has unique charateristics of requiring consistent, but with margin of tolerance of inaccuracy. The current regim of neural network training works but the evaluation of the performance is not straightforward and requires more in-depth knowledge on deep learning models to implement better architecture.

During the course of the project, the various obstacles and problem faced has allowed me to gain a greater understanding of the difficulty in solving self-driving cars and. The detailed examinations of the predictions compared to the target steering shows alarming inaccuracy.

Although, I was able to predict accurately for majority of the highway driving due to the roads turning curvature being less steep than city driving. The model failed terribly at predicting large steering angle below are some recommendations on how to further improve performance and understand/learn about the problem and obstacles faced.

- Adding lane lines detection would enable the neural networks to have boundaries/safety zone, hence, when the prediction requires the vehicle to turn beyond the safety zone the hardcoded logic would step as human drive to take over.
- Edge detection algorithm can also be utilised to further reduce the data size and to provide safety boundries at the same time.
- Additional inputs should be applied like speed, gas throttle and braking. As the function of steering angle are inter-dependent when a human is driving apart from just vision (images).
- Intuitively I feel that different modules of the self-driving car must work in tandem to allow the car to driving safely, an example would be a neural network that detects road surfaces, then detecting lanes and finally predicting the steering angle.
- Maybe, an alternative network architecture to have more constraints on the output of the neural network so that it's less likely to produce opposite steering predictions, such as using multiple output nodes to vote for steering control.

## Setup

###### Required libraries

`numpy`	| `pandas` | `matplotlib` | `opencv` | `h5py` | `imageio` | `sklearn` | `keras`

###### Instructions

1. After downloading, download the source data from the provided link and unpack in their respective folders.

## File Navigation

```
Capstone_Project
|__ datasets
|	|__ camera
|		|__ 2016-01-30--11-24-51.h5
|		...
|	|__ log
|		|__ 2016-01-30--11-24-51.h5
|__ epochs
|	|__ *.h5				## model checkpoints
|__ images
|	|__ *.jpg				## images for notebook
|		...
|__ video
|	|__ *.mp4			## output video to check footage
|		...
|__ 01_eda.ipynb
|__ 02_modeling.ipynb
|__ 03_model_selection.ipynb
|__ LICENSE
|__ README.md

```

## Data sources

Source 1: `Udacity Self Driving Car` [Link](http://academictorrents.com/details/9b0c6c1044633d076b0f73dc312aa34433a25c56)

Source 2: `comma.ai driving dataset` [Link](http://academictorrents.com/details/58c41e8bcc8eb4e2204a3b263cdf728c0a7331eb)


## References

- Udacity Self Driving Challenger [Link](https://github.com/udacity/self-driving-car/tree/master/challenges/challenge-2)
- comma.ai [Link](https://github.com/commaai/research)
- Nvidia End-to-End Deep Learning for Self-Driving Cars [Link](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)