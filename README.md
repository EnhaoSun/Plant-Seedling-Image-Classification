# Plant Seedling image classification

## Aim
This project aims to explore the performance of dif-ferent CNN models in plant seedling image clas-sification.

## Data set
The dataset that used in this researchcontains approximately 960 unit plants belonging to 12 species at several growth stages.
The dataset is hosted by Aarhus Uni-versity [here](https://vision.eng.au.dk/plant-seedlings-dataset/). This dataset is also used for Benchmark of Plant Seedling Classification Algorithms ([Quick link to the paper](https://arxiv.org/abs/1711.05458)

There are 794 images in the test datasetand 4750 images are remained for training and validationdataset. To simulate the test dataset, we choose 794 imagesas our validation dataset which means the training datasethas 3956 images. 

## CNN Models
The CNN models applied in this project includes ALexNet,VGG and GoogLeNet. 
We compare the test ac-curacy of each CNN models in our dataset and the conclusion is GoogLeNet has the best performance with 92 % test accuracy.  We also apply normalization methods like gray scaling andchannel standardization to preprocess the images.However, no significant improvements are detected with these two approaches
