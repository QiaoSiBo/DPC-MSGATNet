# DPC-MSGATNet



Pytorch code for the paper 
["DPC-MSGATNet: Dual-Path Chain Multi-Scale 
Gated Axial-Transformer Network for Four-chamber View Segmentation in Fetal Echocardiography"], which has been accepted by Complex & Intelligent Systems.




### Using the code:

The code is stable using Python 3. 7. 6, Pytorch 1. 7. 1, Torchvision 0. 8. 2, CUDA 11. 0. 221, and CUDNN 8. 0. 5

The DPC-MSGATNet is trained with NVIDIA Telsa V100 32G. If you want to know more detailed information about Hardware and Software, please go to our manuscript, thanks! 


### Links for downloading the Datasets:

1) MoNuSeG Dataset - <a href="https://monuseg.grand-challenge.org/Data/"> Link (Original)</a> 
2) GLAS Dataset - <a href="https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/"> Link (Original) </a> 
3) Fetal US FC view dataset from the paper will be made public soon! But, we have provided several test samples and our pretrained weights of DPC-MSGATNet in the repo. 

### Links for downloading the Pth of DPC-MSGATNet:

The pth of DPC-MSGATNet for fetal US FC view segmentation can be found at here. <a href="https://drive.google.com/file/d/1HCegUCvuTIDNuvu1kCll0jWMHgY77V2x/view?usp=share_link"> DPC-MSGATNet</a>

## Using the Code for your dataset

### Dataset Preparation

Prepare the dataset in the following format for easy use of the code. The train and test folders should contain two subfolders each: img and label. Make sure the images their corresponding segmentation masks are placed under these folders and have the same name for easy correspondance. Please change the data loaders to your need if you prefer not preparing the dataset in this format.



```bash
Train Folder-----
      img----
          0001.png
          0002.png
          .......
      labelcol---
          0001.png
          0002.png
          .......
Validation Folder-----
      img----
          0001.png
          0002.png
          .......
      labelcol---
          0001.png
          0002.png
          .......
Test Folder-----
      img----
          0001.png
          0002.png
          .......
      labelcol---
          0001.png
          0002.png
          .......

```

- The ground truth images should have pixels corresponding to the labels. Example: In case of binary segmentation, the pixels in the GT should be 0 or 255.

### Training Command:

```bash 
python train.py --train_dataset "enter train directory" --val_dataset "enter validation directory" --direc 'path for results to be saved' --batch_size 4 --epoch 405 --save_freq 5 --modelname "msgatnet" --learning_rate 0.001 --imgsize 224 --blocksize 32 --gray "no"
```



### Testing Command:

```bash 
python test.py --loaddirec "./saved_model_path/model_name.pth" --val_dataset "test dataset directory" --direc 'path for results to be saved' --batch_size 1 --modelname "msgatnet" --imgsize 224 --blocksize 32 --gray "no" --mode "multiple"
```




### Acknowledgement:

We refer to the code of <a href="https://github.com/jeya-maria-jose/Medical-Transformer"> Medical-Transformer </a>. The axial attention code is developed from <a href="https://github.com/csrhddlam/axial-deeplab">axial-deeplab</a>. 

# Citation:

```bash
The repo and paper can be cited when the manuscript is accepted for publication. We will continue to update this repository. Please stay tuned!!! 
```

Open an issue or mail me directly in case of any queries or suggestions. 
