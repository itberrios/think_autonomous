# Welcome to the YOLO Workshop!!

YOLO is an algorithm you can find on the website: https://pjreddie.com/
Invented by Joseph Redmon, it is today state of the art in object detection

The goal of this workshop is to learn how to use YOLO quickly to classify objects

The original paper for Yolov3 can be found here: https://arxiv.org/abs/1804.02767.pdf

## Language
All the code will be developed in Python 3

## Requirements
You will need to choose a YOLO configuration, here are the links for 416x416

For 416x416
* CONFIGURATION: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
* WEIGHTS (pretrained): https://pjreddie.com/media/files/yolov3.weights

For other size, simply download by clicking the other links on the YOLO page
https://pjreddie.com/darknet/yolo/

Our goal will be to detect existing objects like cars or people
If you want to detect your own objects, you will have to train it following the procedure on the website
Once your training is finished, you can use your weights
![](https://pjreddie.com/media/image/Screen_Shot_2018-03-24_at_10.48.42_PM.png)


## Colab & Jupyter
We will work with Google Colab and Jupyter Notebook
Colab will allow for simpler installation and possibly the use of GPU
![](https://i0.wp.com/ledatascientist.com/wp-content/uploads/2019/04/jupyter_colab_small_axbdcm.png?resize=502%2C267&ssl=1)

To make your .ipynb file work with Colab, you will need to import the files into your Google Drive account and link the file
Here is the code to link your Google Drive to your colab

```python
import os
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

os.chdir("/content/drive/My Drive/path_to_your_folder")
!ls
```
