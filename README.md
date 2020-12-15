# CoviDNN
Using transfer learning for COVID-19 Detection in X-Rays

## Run Project
```
python3 train.py <MODEL_NAME>
```

## Pre-Trained Models
Pre-Trained models are used with default ImageNet weights.
- VGG16
- VGG19
- ResNet50
- ResNet50V2
- DenseNet121
- InceptionV3
- Xception
- MobileNet

##  Preprocessing

We have used the three following preprocessing methods to improve the performance of the network by enhancing the contrast of our sample x-ray images.
- Adaptive histogram equalization
- Denoising
- Contrast stretching

## Model Architecture
The models were created and trained using Keras with a Tensorflow 2.0 backend. All metrics were calculated using built in functions in Scikit-learn.

### *Control*

To compare the results of using transferling, we built and trained two ‘default’ models from scratch. They had the following network architecture:
- Input - 200x200
- Conv 3x3, 32 kernels, ReLU
- MaxPool 2x2
- Conv 3x3, 64 kernels, ReLU
- MaxPool 2x2
- Conv 3x3, 128 kernels, ReLU
- Dense 128, ReLU
- Output, 1, Sigmoid

These models were initialized with random weights, trained for 100 epochs and used all the same hyperparameters as the pre-trained models. The difference between the two is the use of data augmentation as described above. The naive default model does not use any data augmentation whilst the default model does.

### *Transfer Learning*

To investigate the value of transfer learning, we tested out several different pretrained models. These models are commonly used through the literature. We downloaded the ImageNet weights directly from Keras. For the purpose of feature extraction, we used 8 different models as indicated in the table below.
They all had the following network architecture:
- Input - 200x200
- [Feature Extraction - Pretrained with ImageNet (not trainable)]
- Trainable Dense 128, ReLU
- Output, 1, Sigmoid

All models were compiled to train using Stochastic Gradient Descent (SGD) with a learning rate of 0.001 and a momentum of 0.9 to minimise binary cross-entropy. All models were trained for 100 epochs and validated at the end of each epoch using the given validation set. At the end of training, the accuracy, precision, recall and f1 score were also calculated using the standard formula.

### Sample Training
![Accuracy](/analysis/full_acc_plot.png)
![Loss](/analysis/full_loss_plot.png)
