# Perceptual Losses for Real Time Style Transfer
Pytorch reproduction of the paper ["Perceptual Losses for Real Time Style Transfer"](https://arxiv.org/pdf/1603.08155.pdf "Paper Link"). Some improvements are made by adopting ideas from ["A Learned Representation for Artistic Style"](https://arxiv.org/pdf/1610.07629.pdf)
1. Zero-padding is replaced with mirror-padding. 
2. Transposed convolution is replaced with up-sampling and covolution. 

## Dependencies
```
python 3.6.5
pytorch 0.4.1.post2
```
## Dataset
The residual network is trained with [`LSUN`](http://lsun.cs.princeton.edu/2017/ "LSUN"). 
The pre-trained VGG-net provided by pytorch is used. 

## Usage
```
python main.py
```
You can run your own experiment by giving parameters manually. 

## Results
Original content image (Cape Manzamo, Okinawa, Japan): 

<img src="https://github.com/minkyu-choi04/Perceptual_Losses_for_Real_Time_Style_Transfer/blob/master/content.jpg" alt="Original Content Image" width="500"/>



Style image:

<img src="https://github.com/minkyu-choi04/Perceptual_Losses_for_Real_Time_Style_Transfer/blob/master/style.jpg" alt="Style Image" width="500"/>



Resulting image:

<img src="https://github.com/minkyu-choi04/Perceptual_Losses_for_Real_Time_Style_Transfer/blob/master/sample_output/output_test_e0b8900.jpg" alt="Resulting Image" width="500"/>



Samples from LSUN dataset:

|Original Images|<img src="https://github.com/minkyu-choi04/Perceptual_Losses_for_Real_Time_Style_Transfer/blob/master/sample_output/output_train_gt_e0b4700.jpg" alt="Style Image" width="200"/>|<img src="https://github.com/minkyu-choi04/Perceptual_Losses_for_Real_Time_Style_Transfer/blob/master/sample_output/output_train_gt_e0b3100.jpg" alt="Style Image" width="200"/>|<img src="https://github.com/minkyu-choi04/Perceptual_Losses_for_Real_Time_Style_Transfer/blob/master/sample_output/output_train_gt_e0b2900.jpg?raw=true" alt="Style Image" width="200"/>|
|-------------|-------------|-------------|-------------|
|**Resulting Images**|<img src="https://github.com/minkyu-choi04/Perceptual_Losses_for_Real_Time_Style_Transfer/blob/master/sample_output/output_train_e0b4700.jpg" alt="Style Image" width="200"/>|<img src="https://github.com/minkyu-choi04/Perceptual_Losses_for_Real_Time_Style_Transfer/blob/master/sample_output/output_train_e0b3100.jpg" alt="Style Image" width="200"/>|<img src="https://github.com/minkyu-choi04/Perceptual_Losses_for_Real_Time_Style_Transfer/blob/master/sample_output/output_train_e0b2900.jpg?raw=true" alt="Style Image" width="200"/>|
