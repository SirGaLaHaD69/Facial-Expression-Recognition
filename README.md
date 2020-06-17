# Facial Expression Recogntion (FER)


This Project is a real time facial expression recogniser built using OpenCV 
with the help of Machine Learning  and deep Nets.It shows the most likely
expression of a face like Happy,Angry,Calm, Surprised etc.

### *It basically consists of two sections*

## using KNN (K Nearest  Neighbours) Algorithm.
  This model has achieved  a very high accuarcy on faces like mine and that of
  my friends.The reason being simple, it's trained on a dataset  consisting of
  faces generated through openCV locally on my Computer.
  
  However this model fails to give good results on unseen data ( New faces).

## using CNN ( Convolutional Neural Networks) Algorithm
   This Model has been trained on the Kaggle dataset FER2013
   which had around 38k images. I trained this model on 20k of
   them and achieved an accuracy of 64% ( Early stopping to avoid 
   Overfitting)
   
   This Model works well  on New Faces too.
