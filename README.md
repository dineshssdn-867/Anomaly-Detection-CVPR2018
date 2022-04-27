
# Anomaly Detection

Anomaly detection, also called outlier detection, is the identification of unexpected events, observations, or items that differ significantly from the norm.

# Multiple Instance Learning

In MIL, precise temporal locations of anomalous events in videos are unknown. Video-level labels indicating the presence of an anomaly.

Single video is a bag if the instance of video contains the anomaly we label it as a positive bag(anomalus video) else we consider it negative video(normal video).

# Approach
***
## C3d Architecture
The C3D model is given an input video segment of 16 frames (after downsampling to a fixed size which depends on dataset used) and the outputs a 4096-element vector.
The fully connected layers have a size of 4096 dimensions which will be used in the DNN model for calculating the anomaly score\
![Screenshot 2022-04-26 213236](https://user-images.githubusercontent.com/65583643/165343456-8c2f7475-0476-432e-b4c9-14121bdf21ed.jpg)


## I3d Architecture
The inflated convolution i.e. 3d convolution are performed on the 2D cnn model and after performing number of convolutions on the previous layer and also applying max pooling the results are concated and that result is called an inception module.
The I3D Architecture gives a size of 1024 dimensions which will be used in the DNN model for calculating the anomaly score.\
![1_Ab76Q3eRUOOuIX87hs-GZg](https://user-images.githubusercontent.com/65583643/165343725-d29f6e39-9b62-4f64-937b-418656c17396.png)
![Approach](https://imgur.com/2Qj0lRe)

## DNN Model

Feature of 16 frames clip are represented in the form of (4096D and 1024D) were fed into a 3-layer feed forward neural network. This approach will use forward propagation and backward propagation using hinge loss formulation, sparsity and smoothness.


# Results

## True Positive and False Negative Using C3D

![Explosion041_x264](https://user-images.githubusercontent.com/65583643/165344458-0afe1612-d236-4959-b607-e0b7b25018bb.gif)
![Abuse040_x264](https://user-images.githubusercontent.com/65583643/165344552-ef6d0943-c97e-4473-a9ac-36ade9ef0b73.gif)\
We have trained our model for 4000 iterations, batch size is 32, learning rate is 0.01 and we have got the sum of  hinge-loss, sparsity loss and smoothness loss which is 1.7413.\
![a](https://user-images.githubusercontent.com/65583643/165346735-0c215069-c022-4248-ab28-cf4d7ba2ae54.jpg)



## True Positive and False Negative Using I3D

![Explosion051_x264](https://user-images.githubusercontent.com/65583643/165345321-8c93a410-4a85-4fd8-a2af-821a771b812c.gif)
![Assault048_x264](https://user-images.githubusercontent.com/65583643/165345590-d53a0fe6-1174-44fb-875a-0cad3b0b4b54.gif)\
We have trained our I3d model for 10000 iterations, batch size is 32, learning rate is 0.01 and we have got the sum of  hinge-loss, sparsity loss and smoothness loss which is 2.23.\
![Screenshot 2022-04-26 214938](https://user-images.githubusercontent.com/65583643/165346589-3f7bb511-3fea-4a31-a4ad-84dd8e5aee61.jpg)

## Conclusion


The I3d Trained model gives results with more accuracy then the results generated using the C3D model.
