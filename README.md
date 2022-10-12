# Image classification and Object Detection

This work presents two examples of using transfer-learning to do image classification and object detection

**`Image Classification and object detection.ipynb`** uses [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, which contains image with size `(28, 28, 1)` and classes of 10 (num 0 to 9)

Work in this Code:
- classify the main subject in an image
- localize it by drawing bounding boxes around it.

example of drawing a box around a image, (original, with_bounding_ox)
<p float="left">
  <img src='original_num.png' width="200" height="200"/>
  <img src='boudning_box_num.png' width="200" height="200"/> 
</p>

and the final result can be evaluated by the iou parameter.
 <img src='result.png' width="2000" height="200"/>
 
 **`Predicting_bounding_box_Caltech_Bird.ipynb`** uses `caltech_birds2010` dataset in the `tensorflow_datasets`. 
 The main work in this script is to predict the location of bounding box using the **MobileNet V2** network and fine-tune the model by moving the top layers. 
 The result:
 
<img src='result_bird.png' width="2000" height="200"/>
