# *Omnidirectional Image Quality Captioning: A Large-scale Database and a New Model*

Pytorch implementation of the paper "Omnidirectional Image Quality Captioning: A Large-scale Database and a New Model"

## :seedling:OIQ-10K Database

### Introduction
OIQ-10K database contains 10,000 omnidirectional images with homogeneous and heterogeneous distortion, which demonstrated by four distortion ranges: no perceptibly distorted region (2,500), one distorted region (2,508), two distorted regions (2,508), and global distortion (2,484). MOS is 1~3.

<p align="center"><img src="https://github.com/WenJuing/IQCaption360/blob/main/imgs/database.png" width="900"></p>

Visualization of omnidirectional images with different distorted regions in the proposed OIQ-10K database. The distortion region(s) of the visual examples in (b) and (c) are marked in red for better visual presentation.

### Distortion Composition

<table style="width: 100%"><thead>
  <tr>
    <th style="width: 25%">Distortion range</th>
    <th style="width: 25%">Distortion type</th>
    <th style="width: 25%">Distortion level</th>
    <th style="width: 25%">Number</th>
    <th style="width: 25%">Remark</th>
  </tr></thead>
<tbody>
  <tr>
    <td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;R1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td>invalid</td>
    <td>invalid</td>
    <td>2,500</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="4">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;R2/R3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td>Gaussian noise</td>
    <td>1~3</td>
    <td>627/627</td>
    <td></td>
  </tr>
  <tr>
    <td>Gaussian blur</td>
    <td>1~3</td>
    <td>627/627</td>
    <td></td>
  </tr>
  <tr>
    <td>Stitching</td>
    <td>1~3</td>
    <td>627/627</td>
    <td></td>
  </tr>
  <tr>
    <td>brightness discontinuity</td>
    <td>1~3</td>
    <td>627/627</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="7">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;R4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</td>
    <td>compression</td>
    <td>-</td>
    <td>1,436</td>
    <td>include: JPEG compression (590), JPEG2000 compression (212), AVC compression (176), HEVC compression (383), VP9 compression (75)</td>
  </tr>
  <tr>
    <td>Guassian noise</td>
    <td>-</td>
    <td>248</td>
    <td></td>
  </tr>
  <tr>
    <td>Guassian blur</td>
    <td>-</td>
    <td>155</td>
    <td></td>
  </tr>
  <tr>
    <td>Stitching</td>
    <td>-</td>
    <td>75</td>
    <td></td>
  </tr>
  <tr>
    <td>downsampling</td>
    <td>-</td>
    <td>75</td>
    <td></td>
  </tr>
  <tr>
    <td>JPEG XT and TM</td>
    <td>-</td>
    <td>319</td>
    <td></td>
  </tr>
  <tr>
    <td>Authentic distortion</td>
    <td>invalid</td>
    <td>176</td>
    <td></td>
  </tr>
</tbody></table>

* R1: no perceptibly distorted region, R2: one distorted region, R3: two distorted regions, R4: global distortion
* The MOS and more detailed information about images see the file <a href="https://drive.google.com/file/d/1D13S2JuMt636nImaJzo_4IYBE30K6r0U/view?usp=sharing" target="_blank">data_info.csv</a>

### Database Download
Click here: <a href="https://pan.baidu.com/s/1Uy0AR9B2oCAIJuLCuZEtLg" target="_blank">https://pan.baidu.com/s/1Uy0AR9B2oCAIJuLCuZEtLg (pass: jvga)</a> to downloard the OIQ-10K database.

## :dart:IQCaption360 Architecture

<p align="center"><img src="https://github.com/WenJuing/IQCaption360/blob/main/imgs/IQCaption360.png" width="900"></p>
The architecture of the proposed IQCaption. It contains four parts: (a) backbone, (b) adaptive feature aggregation module, (c) distortion range prediction network, and (d) quality score prediction network.

### Textual Output
The Caption360 can output a caption to represent the perceptual quality of omnidirectional images.
* Example1
  
<p><img src="https://github.com/WenJuing/IQCaption360/blob/main/imgs/output1.jpg" width="700"></p>

* Example2
  
<p><img src="https://github.com/WenJuing/IQCaption360/blob/main/imgs/output2.jpg" width="700"></p>

* Example3
  
<p><img src="https://github.com/WenJuing/IQCaption360/blob/main/imgs/output3.jpg" width="700"></p>

* Example4
  
<p><img src="https://github.com/WenJuing/IQCaption360/blob/main/imgs/output4.jpg" width="700"></p>

## :eyes:Usage

### Install
1. Clone this repo:
```python
git clone https://github.com/WenJuing/IQCaption360
cd IQCaption360
``` 
2. Create an Anaconda environment with <a href="https://shi-labs.com/natten/" target="_blank">natten 0.14.6</a>

### Inference one Image
```python
CUDA_VISIBLE_DEVICES=0 python inference_one_image.py --load_ckpt_path /home/xxxy/tzw/IQCaption360/ckpt/IQCaption_OIQ-10K.pth --test_img_path /home/xxxy/tzw/databases/viewports_8/ref2.jpg
``` 
* Download `weights` [ <a href="https://drive.google.com/file/d/1UukN1kKtPkO-2a4ITn3-_d4KD8Dp81oM/view?usp=sharing" target="_blank">google drive</a> | <a href="https://pan.baidu.com/s/1bp4dxKpReAVp8cszuDysng" target="_blank">baidu cloud</a> (password: jeop) ] pretrained on OIQ-10K database
* `test_img` offers a group of processed omnidirectional images

### Train and Test

Edit `config.py` for configuration

* Train

```python
CUDA_VISIBLE_DEVICES=0 python train.py
```
or
```python
sh run.sh
```

* Test

```python
CUDA_VISIBLE_DEVICES=0 python test.py
```

## Citation
```plaintext
@article{yan2024caption360,
title={Omnidirectional image quality captioning: A large-scale database and a new model},
author={Yan, Jiebin and Tan, Ziwen and Fang, Yuming and Chen, Junjie and Wang, Zhou},
year={2024}
}
```
