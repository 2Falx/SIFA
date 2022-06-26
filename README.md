# Image Style Transfer for Brain Vessel Segmentation from multi-modal MRI
### EURECOM SEMESTER PROJECT - SPRING 2022
### Supervisors: Professor M. Zuluaga - F. Galati

## Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation

Tensorflow v2 implementation of the SIFA unsupervised cross-modality domain adaptation framework. <br/>
Please refer to the branch [SIFA-v1](https://github.com/cchen-cc/SIFA/tree/SIFA-v1) for the original paper and code <br/>

## Original Paper
[Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation](https://arxiv.org/abs/2002.02255)
<br/>
IEEE Transactions on Medical Imaging
<br/>
<br/>
<p align="center">
  <img src="Network.png">
</p>

## How to run
* Open and follow the "SIFA_Implementation_Tf2-.ipynb" jupyter notebook modifying the indicated variable and parameters to adapt it to your dataset.
* Set the home_path in cell #3 (here it corresponds to the same folder of the above mentioned notebook).
* Store your data inside a folder "data" in the home_path
* Store SWI images in /data/SWI and TOF images in /data/TOF (You can find an example folder for a single 3D image in cell #6)
* Select the wanted Spacing in cell #13 to allow images resizing (here SWI Spacing is chosen)
* Cell #23 contains all the useful statistics for OUR dataset: comment the last line to use the extracted ones from your data
* Slices are preprocessed in Cell #31 and transformed into Tfrecord in Cell #34 (Here you can modify the "tfrecords_folder" name)
* Modify the "percentage" parameter in the "split_file" function (Cell #35) to select the wanted train/validation split percentage
* 


## Data Preparation

## Train

## Acknowledgement
The code is a revisiting version of the original [SIFA Implementation](https://github.com/cchen-cc/SIFA/tree/SIFA-v1)
Part of the code is revised from the [Tensorflow implementation of CycleGAN](https://github.com/leehomyc/cyclegan-1).

