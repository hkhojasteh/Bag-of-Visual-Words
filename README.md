# Bag-of-Visual-Words-OpenCV
Implementation of a content based image classifier using the [bag of visual words model][1] in C++ with OpenCV.

The program will generate a visual vocabulary and train a classifier using a user provided set of already classified images. After the learning phase, we will use the generated vocabulary and the trained classifier to predict the class for any image given to the script by the user.

The learning consists of:

1. Extracting local features of all the dataset image with SIFT feature extractor
2. Generating a codebook of visual words with clustering of the features
3. Aggregating the histograms of the visual words for each of the traning images
4. Feeding the histograms to the SVM classifier to train a model

This code relies on:

 - SIFT features for local features
 - k-means for generation of the words via clustering
 - SVM as classifier using the OpenCV library

The folder can have any name. One example dataset would be the [Caltech 101 dataset][2].

### References:
#### OpenCV:
Open Source Computer Vision Library https://github.com/opencv
#### SIFT:
David G. Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision, 60, 2 (2004), pp. 91-110.

[1]: https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision
[2]: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
