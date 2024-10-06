# *Omnidirectional Image Quality Captioning: A Large-scale Database and a New Model*

Pytorch implementation of the paper "Omnidirectional Image Quality Captioning: A Large-scale Database and a New Model"

# OIQ-10K database

## Introduction
OIQ-10K database contains 10,000 omnidirectional images with homogeneous and heterogeneous distortion, which demonstrated by four distortion ranges: no perceptibly distorted region (2,500), one distorted region (2,508), two distorted regions (2,508), and global distortion (2,484). MOS is 1~3.

<p align="center"><img src="https://github.com/WenJuing/IQCaption360/blob/main/imgs/database.png" width="900"></p>

Visualization of omnidirectional images with different distorted regions in the proposed OIQ-10K database. The distortion region(s) of the visual examples in (b) and (c) are marked in red for better visual presentation.

## Detailed information

<table style="width: 100%"><thead>
  <tr>
    <th style="width: 25%">Distortion range</th>
    <th style="width: 25%">Distortion type</th>
    <th style="width: 25%">Number</th>
    <th style="width: 25%">Remark</th>
  </tr></thead>
<tbody>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;R1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td>-</td>
    <td>2,500</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="4">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;R2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td>Gaussian noise</td>
    <td>627</td>
    <td></td>
  </tr>
  <tr>
    <td>Gaussian blur</td>
    <td>627</td>
    <td></td>
  </tr>
  <tr>
    <td>Stitching</td>
    <td>627</td>
    <td></td>
  </tr>
  <tr>
    <td>brightness discontinuity</td>
    <td>627</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="4">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;R3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td>Gaussian noise</td>
    <td>627</td>
    <td></td>
  </tr>
  <tr>
    <td>Gaussian blur</td>
    <td>627</td>
    <td></td>
  </tr>
  <tr>
    <td>Stitching</td>
    <td>627</td>
    <td></td>
  </tr>
  <tr>
    <td>brightness discontinuity</td>
    <td>627</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="7">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;R4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td>compression</td>
    <td>1,436</td>
    <td>JPEG compression (590), JPEG2000 compression (212), AVC compression (176), HEVC compression (383), VP9 compression (75)</td>
  </tr>
  <tr>
    <td>Guassian noise</td>
    <td>248</td>
    <td></td>
  </tr>
  <tr>
    <td>Guassian blur</td>
    <td>155</td>
    <td></td>
  </tr>
  <tr>
    <td>Stitching</td>
    <td>75</td>
    <td></td>
  </tr>
  <tr>
    <td>downsampling</td>
    <td>75</td>
    <td></td>
  </tr>
  <tr>
    <td>JPEG XT and TM</td>
    <td>319</td>
    <td></td>
  </tr>
  <tr>
    <td>Authentic distortion</td>
    <td>176</td>
    <td></td>
  </tr>
</tbody></table>


## Download
Click here: <a href="https://drive.google.com/drive/folders/18vCXea59S9JMYSaXBAe82mxa-_6i7FFJ" target="_blank">Google drive</a>

The data_info.csv contains 

# IQCaption360 Architecture

<p align="center"><img src="https://github.com/WenJuing/IQCaption360/blob/main/imgs/IQCaption360.png" width="900"></p>
The architecture of the proposed IQCaption. It contains four parts: (a) backbone, (b) adaptive feature aggregation module, (c) distortion range prediction network, and (d) quality score prediction network.


## Train and Test
* The pre-trained weights can be downloaded at the <a href="https://drive.google.com/drive/folders/18vCXea59S9JMYSaXBAe82mxa-_6i7FFJ" target="_blank">Google drive</a>
* Edit the `config.py` for an implement
* Run the file `train.py` and `test.py` for training and testing
* If you need train our model on other databases, loading weights pre-trained on JUFE could has better training results

## Textual Output
The Caption360 can output a caption to represent the perceptual quality of omnidirectional images.
<p align="center"><img src="https://github.com/WenJuing/IQCaption360/blob/main/imgs/output_example.png" width="600"></p>


# Citation
```plaintext
@article{yan2024caption360,
title={Omnidirectional image quality captioning: A large-scale database and a new model},
author={Yan, Jiebin and Tan, Ziwen and Fang, Yuming and Chen, Junjie and Wang, Zhou},
year={2024}
}
```
