# *Omnidirectional Image Quality Captioning: A Large-scale Database and a New Model*

Jiebin Yan<sup>1</sup>, Ziwen Tan<sup>1</sup>, Yuming Fang<sup>1</sup>, Junjie Chen<sup>1</sup>, Wenhui Jiang<sup>1</sup>, and Zhou Wang<sup>2</sup>.

<sup>1</sup> School of Computing and Artificial Intelligence, Jiangxi University of Finance and Economics

<sup>2</sup> Department of Electrical and Computer Engineering, University of Waterloo

## :four_leaf_clover:News:

- **February 21, 2025**: The arXiv version of our paper is released: <a href="https://arxiv.org/abs/2502.15271" target="_blank">https://arxiv.org/abs/2502.15271.</a>

- **January 27, 2025**: Our paper is accepted by *IEEE T-IP*!

- **October 6, 2024**: We upload the OIQ-10K database and the IQCaption360 code.

## :seedling:OIQ-10K Database

### Introduction
OIQ-10K database contains 10,000 omnidirectional images with homogeneous and heterogeneous distortion, which demonstrated by four distortion situation: no perceptibly distorted region (2,500), one distorted region (2,508), two distorted regions (2,508), and global distortion (2,484). MOS is 1~3.

<p align="center"><img src="https://github.com/WenJuing/IQCaption360/blob/main/imgs/database.jpg" width="900"></p>

Visualization of omnidirectional images with different distorted regions in the proposed OIQ-10K database. The distortion region(s) of the visual examples in (b) and (c) are marked in red for better visual presentation.

### Establish Details

<table style="width: 100%"><thead>
  <tr>
    <th style="width: 20%">Situation</th>
    <th style="width: 20%">Description</th>
    <th style="width: 10%">Coarse stage</th>
    <th style="width: 20%">Refine technique</th>
    <th style="width: 30%">Refinement stage</th>
  </tr></thead>
<tbody>
  <tr>
    <td>CnoDist</td>
    <td>No perceptibly distorted region</td>
    <td>3903=1001 (OIQA databases) +2498 (Flickr)+404 (Pixexid)</td>
    <td>Deduplication, <a href="https://github.com/bbonik/distributional_dataset_undersampling" target="_blank">database shaping technique</a></td>
    <td>2500</td>
  </tr>
  <tr>
    <td>CdistR1</td>
    <td>One distorted region</td>
    <td>3096=258x4x3 (extend from JUFE)</td>
    <td>Manual selection</td>
    <td>2508=209x4x3</td>
  </tr>
  <tr>
    <td>CdistR2</td>
    <td>Two distorted regions</td>
    <td>3096=258x4x3 (extend from JUFE)</td>
    <td>Manual selection</td>
    <td>2508=209x4x3</td>
  </tr>
  <tr>
    <td>CdistGl</td>
    <td>Global distortion</td>
    <td>2484=2071 (OIQA databases)+237 (Flickr)+176 (Pixexid)</td>
    <td>All save</td>
    <td>2484</td>
  </tr>
  <tr>
    <td>Total</td>
    <td>-</td>
    <td>12,579</td>
    <td>-</td>
    <td>10,000</td>
  </tr>
</tbody>
</table>

* CnoDist: no perceptibly distorted region, CdistR1: one distorted region, CdistR2: two distorted regions, CdistGl: global distortion

### Distortion Composition

<table style="width: 100%"><thead>
  <tr>
    <th style="width: 25%">Situation</th>
    <th style="width: 25%">Distortion type</th>
    <th style="width: 25%">Distortion level</th>
    <th style="width: 25%">Number</th>
    <th style="width: 25%">Remark</th>
  </tr></thead>
<tbody>
  <tr>
    <td>CnoDist</td>
    <td>invalid</td>
    <td>invalid</td>
    <td>2,500</td>
    <td>source: JUFE (115), CVIQ (10), OIQA (10), Salient360! (60), Xu2021 (436), NBU-SOID (7), LIVE 3D VR IQA (12), Pixexid (150), Flickr (1,700)</td>
  </tr>
  <tr>
    <td rowspan="4">CdistR1/CdistR2</td>
    <td>Gaussian noise</td>
    <td>1~3</td>
    <td>627/627</td>
    <td>source: extend from JUFE</td>
  </tr>
  <tr>
    <td>Gaussian blur</td>
    <td>1~3</td>
    <td>627/627</td>
    <td>source: extend from JUFE</td>
  </tr>
  <tr>
    <td>Stitching</td>
    <td>1~3</td>
    <td>627/627</td>
    <td>source: extend from JUFE</td>
  </tr>
  <tr>
    <td>brightness discontinuity</td>
    <td>1~3</td>
    <td>627/627</td>
    <td>source: extend from JUFE</td>
  </tr>
  <tr>
    <td rowspan="7">CdistGl</td>
    <td>Compression</td>
    <td>-</td>
    <td>1,436</td>
    <td>include: JPEG compression (590), JPEG2000 compression (212), AVC compression (176), HEVC compression (383), VP9 compression (75)</td>
  </tr>
  <tr>
    <td>Gaussian noise</td>
    <td>-</td>
    <td>248</td>
    <td>source: LIVE 3D VR (75), OIQA (78), Flickr (95)</td>
  </tr>
  <tr>
    <td>Gaussian blur</td>
    <td>-</td>
    <td>155</td>
    <td>source: LIVE 3D VR (75), OIQA (80)</td>
  </tr>
  <tr>
    <td>Stitching</td>
    <td>1~5</td>
    <td>75</td>
    <td>source: LIVE 3D VR</td>
  </tr>
  <tr>
    <td>downsampling</td>
    <td>1~5</td>
    <td>75</td>
    <td>source: LIVE 3D VR</td>
  </tr>
  <tr>
    <td>JPEG XT and TM</td>
    <td>-</td>
    <td>319</td>
    <td>source: NBU-HOID</td>
  </tr>
  <tr>
    <td>Authentic distortion</td>
    <td>invalid</td>
    <td>176</td>
    <td>source: Pixexid</td>
  </tr>
</tbody></table>

* The MOS and more detailed information about images see the file <a href="https://drive.google.com/file/d/1IJgsXB0GcavodXsEa6ee003s5WVp8WtJ/view?usp=sharing" target="_blank">OIQ-10K_data_info.csv</a>

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
@article{yan2025omnidirectional,
  title={Omnidirectional image quality captioning: A large-scale database and a new model},
  author={Yan, Jiebin and Tan, Ziwen and Fang, Yuming and Chen, Junjie and Jiang, Wenhui and Wang, Zhou},
  journal={IEEE Transactions on Image Processing},
  year={2025},
  volume={34},
  pages={1326-1339},
}
```
