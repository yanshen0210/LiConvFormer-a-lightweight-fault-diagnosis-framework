# LiConvFormer: A lightweight fault diagnosis framework using separable multiscale convolution and broadcast self-attention
* Core codes for the paper:
<br> [LiConvFormer: A lightweight fault diagnosis framework using separable multiscale convolution and broadcast self-attention](https://www.sciencedirect.com/science/article/pii/S0957417423018407)
* Created by Shen Yan, Haidong Shao, Jie Wang, Xinyu Zheng, Bin Liu.
* Journal: Expert Systems With Applications
  
<div align="center">
<img src="https://github.com/yanshen0210/LiConvFormer-a-lightweight-fault-diagnosis-framework/blob/main/framework.jpg" width="600" />
</div>

## Our operating environment
* Python 3.8
* pytorch  1.10.1
* numpy  1.22.0 (If you get an error when saving data, try lowering your numpy version!)
* and other necessary libs

## Datasets
* [Case1: XJTU gearbox](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing)
* [Case2: XJTU spurgear](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing)
* [Case3: OU bearing](https://drive.google.com/file/d/1PQnIBKzAu098SAl3DUw0n8AHONynpdb7/view?usp=sharing)
* [Save dataset](https://drive.google.com/file/d/10XQDVN9YqbM7--X3dB55Io1eRLsLmruI/view?usp=sharing)

## Guide 
* This repository provides a lightweight fault diagnosis framework. 
* It includes the pre-processing for the data and the model proposed in the paper. 
* We have also integrated 7 baseline methods including 4 CNN methods and 3 fault diagnosis methods based on CNN-Transformer for comparison.
* `train_val_test.py` is the train&val&test process of all methods.
* You need to load the data in above Datasets link at first, and put them in the `data` folder. Then run in `args_diagnosis.py`
<br> Pay attention to that if you want to run the data pre-process, you need to load [Case1](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing),
[Case2](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing) and [Case3](https://drive.google.com/file/d/1PQnIBKzAu098SAl3DUw0n8AHONynpdb7/view?usp=sharing) in Datasets,
<br> and set --save_dataset (in `args_diagnosis.py`) to True; or you can just load the [Save dataset](https://drive.google.com/file/d/10XQDVN9YqbM7--X3dB55Io1eRLsLmruI/view?usp=sharing), and set --save_dataset to False.
* You can also choose the modules or adjust the parameters of the model to suit your needs.

## Initial learning rate
* Liconvformer: Case1--0.01;  Case2--0.001;  Case3--0.01
* CLFormer: Case1--0.01;  Case2--0.001;  Case3--0.01
* convoformer_v1_small: Case1--0.001;  Case2--0.001;  Case3--0.001
* mcswint: Case1--0.001;  Case2--0.001;  Case3--0.01
* MobileNet: Case1--0.01;  Case2--0.001;  Case3--0.001
* MobileNetV2: Case1--0.01;  Case2--0.001;  Case3--0.001
* ResNet18: Case1--0.001;  Case2--0.001;  Case3--0.001
* MSResNet: Case1--0.001;  Case2--0.001;  Case3--0.001

## Pakages
* `data` needs loading the Datasets in above links
* `datasets` contians the pre-processing process for the data
* `models` contians 8 methods including the proposed method
* `utils` contians train&val&test processes

## Citation
If our work is useful to you, please cite the following paper, it is the greatest encouragement to our open source work, thank you very much!
```
@paper{
  title = {LiConvFormer: A lightweight fault diagnosis framework using separable multiscale convolution and broadcast self-attention},
  author = {Shen Yan, Haidong Shao, Jie Wang, Xinyu Zheng, Bin Liu},
  journal = {Expert Systems With Applications},
  volume = {237, Part A},
  pages = {121338},
  year = {2023},
  doi = {doi.org/10.1016/j.eswa.2023.121338},
  url = {https://www.sciencedirect.com/science/article/pii/S0957417423018407},
}
```
