# Playing to the Strengths of High- and Low-Resolution Cues for Ultra-High Resolution Image Segmentation
Qi Li, Jiexin Luo, Chunxiao Chen, Jiaxin Cai, Wenjie Yang, Yuanlong Yu, Shengfeng He, and Wenxi Liu  
Accepted to [IEEE RA-L 2025](https://ieeexplore.ieee.org/abstract/document/11034711)
## Abstract
In ultra-high resolution image segmentation task for robotic platforms like UAVs and autonomous vehicles, existing paradigms process a downsampled input image through a deep network and the original high-resolution image through a shallow network, then fusing their features for final segmentation. Although these features are designed to be complementary, they often contain redundant or even conflicting semantic information, which leads to blurred edge contours, particularly for small objects. This is especially detrimental to robotics applications requiring precise spatial awareness. To address this challenge, we propose a novel paradigm that disentangles the task into two independent subtasks concerning high- and low-resolution inputs, leveraging high-resolution features exclusively to capture low-level structured details and low-resolution features for extracting semantics. Specifically, for the high-resolution input, we propose a region-pixel association experts scheme that partitions the image into multiple regions. For the low-resolution input, we assign compact semantic tokens to the partitioned regions. Additionally, we incorporate a high-resolution local perception scheme with an efficient field-enriched local context module to enhance small object recognition in case of incorrect semantic assignment. Extensive experiments demonstrate the state-of-the-art performance of our method and validate the effectiveness of each designed component.
![framework](https://github.com/liqiokkk/HLRC/blob/main/framework.png)
## Test and train
python==3.7, pytorch==1.10.0, and mmcv==1.7.0 
### dataset
Please download [Cityscapes](https://www.cityscapes-dataset.com/) dataset.  
Please download [Inria Aerial](https://project.inria.fr/aerialimagelabeling/) dataset and [DeepGlobe](https://competitions.codalab.org/competitions/18468) dataset, we follow [FCtL](https://github.com/liqiokkk/FCtL) to split two datasets.  
Create folder named 'InriaAerial', its structure is 
```
    InriaAerial/
    ├── imgs
       ├── train
          ├── xxx_sat.tif
          ├── ...
       ├── test
       ├── val
    ├── labels
       ├── train
          ├── xxx_mask.png(two values:0-1)
          ├── ...
       ├── test
       ├── val
```
Create folder named 'DeepGlobe', its structure is
```
    DeepGlobe/
    ├── img_dir
       ├── train
          ├── xxx_sat.jpg
          ├── ...
       ├── val
       ├── test
    ├── rgb2id
      ├── train
          ├── xxx_mask.png(0-6)
          ├── ...
      ├── val
      ├── test
```
### test
Please download our pre-trained model:  
Baidu Netdisk: https://pan.baidu.com/s/1BllXL6E8B9r7zl77VipcvQ?pwd=651v  Extraction Code: 651v
```
python ./test.py configs/swin/swin_tiny_p4w7_1024x1024_160k_city_pretrain_224x224_1K.py city.pth --eval mIoU
python ./test.py configs/swin/swin_tiny_p4w7_1024x1024_80k_IA_pretrain_224x224_1K.py ia.pth --eval mIoU
python ./test.py configs/swin/swin_tiny_p4w7_768x768_80k_deepglobe_pretrain_224x224_1K.py deepglobe.pth --eval mIoU
```
### train
Please download [Swin Transformer](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth) and  [EfficientViT-B2](https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_cls/README.md#pretrained-efficientvit-classification-models) pre-trianed model.
```
python ./train.py configs/swin/swin_tiny_p4w7_1024x1024_160k_city_pretrain_224x224_1K.py
python ./train.py configs/swin/swin_tiny_p4w7_1024x1024_80k_IA_pretrain_224x224_1K.py
python ./train.py configs/swin/swin_tiny_p4w7_768x768_80k_deepglobe_pretrain_224x224_1K.py
```
## Citation
If you use this code or our results for your research, please cite our papers.
```
@article{li2025playing,
  title={Playing to the Strengths of High-and Low-Resolution Cues for Ultra-high Resolution Image Segmentation},
  author={Li, Qi and Luo, Jiexin and Chen, Chunxiao and Cai, Jiaxin and Yang, Wenjie and Yu, Yuanlong and He, Shengfeng and Liu, Wenxi},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```
For the details of our previous work, please refer to [FCtL](https://github.com/liqiokkk/FCtL) and [SGHRQ](https://github.com/liqiokkk/SGHRQ).
