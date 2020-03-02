# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree]

## Decscription

The goal of this project is to write a software pipeline to detect vehicles in a video. 

## Details

Steps followed are:


### Dataset

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

I have included dataset used in the repo. Here is the count of the dataset.

```
Vehicle train images count: 8792
Non-vehicle train image count: 8968

```
### Feature extraction

There are various feature extraction techniques has been used to train the classifier to detect the cars efficiently.

#### Spatial Binning

While it could be cumbersome to include three color channels of a full resolution image, we can perform spatial binning on an image and still retain enough information to help in finding vehicles. Even going all the way down to 32 x 32 pixel resolution, the car itself is still clearly identifiable by eye, and this means that the relevant features are still preserved at this resolution. A convenient function for scaling down the resolution of an image is OpenCV's cv2.resize(). If you then wanted to convert this to a one dimensional feature vector, numpy's ravel() function can be used.


### Color histogram

In photography a histogram is simply a graphical representation of the number of pixels in the image that fall within a certian range, either luminance or color. For example for a normal luminance histogram the graph shows the number of pixels for each luminance or brightness level from black to white. The higher the peak on the graph the more pixels are at that luminance level. With a color histogram the principle is the same but instead of seeing the levels of black graphed you will now see the number of pixels for each of the three main colors.

A color histogram is a simply a histogram that shows the color level for each individual RGB color channel. 

If we had to, we could differentiate the two images based on the differences in histograms alone. Differentiating images by the intensity and range of color they contain can be helpful for looking at car vs non-car images.


### Histogram of oriented gradients(HOG)

A feature descriptor is a representation of an image or an image patch that simplifies the image by extracting useful information and throwing away extraneous information.

The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image.

In the HOG feature descriptor, the distribution ( histograms ) of directions of gradients ( oriented gradients ) are used as features. Gradients ( x and y derivatives ) of an image are useful because the magnitude of gradients is large around edges and corners ( regions of abrupt intensity changes ) and we know that edges and corners pack in a lot more information about object shape than flat regions.

### Choosing the parameters

Next one is to choose the right parameters to train the classifier to predict the image, I have defined the parameter class to define these parameters.

Here is the choosen parameters.

```
Parameters(
        color_space = 'YCrCb',
        spatial_size = (16, 16),
        orient = 8,
        pix_per_cell = 8,
        cell_per_block = 2,
        hog_channel = 'ALL',
        hist_bins = 32,
        scale = 1.5,
        spatial_feat=True, 
        hist_feat=True, 
        hog_feat=True
)
```

Feature extraction varies with parameters. I have done feature extraction with various parameter combination and I found the above parameter combination is best suited for the feature extraction.


### Classifier

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection. I have decided to use LinearSVC as classifier this project. 

The cars and nor-cars images are loaded and extracted features using the above feature extraction techniques. These values are stored in car_features and notcar_features.We assumes that output of the car_features will be one and notcar_features is zero. 'y' value is filled based on that assumption. 'x' is the combination car_features and notcar_features list.

The StandardScaler assumes data is normally distributed within each feature and will scale them such that the distribution is now centred around 0, with a standard deviation of 1. 'x' values are transformed using the function and get the output scaled_X.

There are few helping libraries to split the dataset. 'train_test_split' funtion from 'sklearn' is one of them which help to split the dataset into train an test data for the classifier.

Finally LinearSVC is defined and pass the x_train and y_train for trainning the classifier. Once it's completed, test datset is used to check the accuracy of the classifier.

Classifier accuracy is around 99 percent on the test split of the dataset.

```
Using: 8 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 5568
3.04 Seconds to train SVC...
Test Accuracy of SVC =  0.9953

```


### Sliding Window 

In the context of computer vision (and as the name suggests), a sliding window is rectangular region of fixed width and height that “slides” across an image. For each of these windows, we would normally take the window region and apply an image classifier to determine if the window has an object that interests us.

Here overlapping detections exist for each of the two vehicles, and in two of the frames, there is a false positive detection on the middle of the road. So in order to combine overlapping detections and remove false positives, heatmap and threahold limit are used. 

### The HOG Sub-sampling

The hog sub-sampling is more efficient method for doing the sliding window approach. The code only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows. The hog sub-sampling helps to reduce calculation time for finding HOG features and thus provided higher throughput rate.

I have decided to choose stating position of the window search from 350px to 656px and cells_per_step reduced to one to get more accurate result.

As explained above, same heatmap and threshold with limit 1 techniqueue is used to combine overlapping detections and remove false positives.


### Pipeline video

Finally create the pipeline vide by processing the each frame of the image with above techniques and create the video out of the processed frames. 

Video:

https://youtu.be/Vyo_xwigQW0 

### Discussion

* Even thpough my code is detecting the vehicle accurately. Sometimes it detects other parts of the road as vehicle as well. I have to improve this by trainining my classifer with huge dataet or would like to use Deep Learning approach.

* My project is depend on on Spatial and Hog Features. It's not giving all the time better result as it detected non-vehicle images as vehicle. I would like to use more feature extraction technique to increase the accuracy.
