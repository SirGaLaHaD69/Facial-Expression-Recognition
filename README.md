Made by <b><i>RITIK PARIDA</i></b>
<br>
CSE,NITR-22
<br>
(This is <b>NOT</b> a Coursera-Guided Project)
# Facial Expression Recogntion (FER)


This Project is a real time facial expression recogniser built using OpenCV 
with the help of Machine Learning  and deep Nets.It shows the most likely
expression of a face like Happy,Angry,Calm, Surprised etc.

### *This Repository contains two models*

## Using KNN (K Nearest  Neighbours) Algorithm.
-  This model has achieved  a very high accuarcy on faces like mine and that of
  my friends.The reason being simple, it's trained on a dataset  consisting of
  faces generated through openCV locally on my Computer.
  
- However this model fails to give good results on unseen data ( New faces).
  

## Using CNN ( Convolutional Neural Networks) Algorithm
-  This Model has been trained on the Kaggle dataset FER2013
   which had around 38k images.Trained on 24k images , validated
   on 6k images. Finally tested on 8k faces.
   
 -  This Model works well  on New Faces too.
 - ### *Model in figures*
    - Val accuracy : 70.5 %
    - Test accuracy ~ 70 %
    - Model that won the Kaggle contest had an acc of 71%
