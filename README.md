# LiConvFormer: A lightweight fault diagnosis framework using separable multiscale convolution and broadcast self-attention
* Core codes for the paper: "LiConvFormer: A lightweight fault diagnosis framework using separable multiscale convolution and broadcast self-attention"(under review)

<div align="center">
<img src="https://github.com/yanshen0210/LiConvFormer-a-lightweight-fault-diagnosis-framework/blob/main/framework.jpg" width="600" />
</div>

## Our operating environment
* Python 3.8
* pytorch  1.10.1
* and other necessary libs

## Guide 
* This repository provides a lightweight fault diagnosis framework. 
* It includes the pre-processing for the data and the model proposed in the paper. 
* We have also integrated 7 baseline methods including 4 CNN methods and 3 fault diagnosis methods based on CNN-Transformer for comparison.
* `train_val_test.py` is the train&val&test process of all methods.
* You need to load the data in following Datasets link at first, and put them in the `data` folder. Then run in `args_diagnosis.py`
* You can also choose the modules or adjust the parameters of the model to suit your needs.

## Datasets
* [Case1: XJTU gearbox](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing)
* [Case2: XJTU spurgear](https://drive.google.com/drive/folders/1ejGZu9oeL1D9nKN07Q7z72O8eFrWQTay?usp=sharing)
* [Case3: OU bearing](https://drive.google.com/file/d/1PQnIBKzAu098SAl3DUw0n8AHONynpdb7/view?usp=sharing)


## Pakages
* `data` needs loading the Datasets in above links
* `datasets` contians the pre-processing process for the data
* `models` contians 8 methods including the proposed method
* `utils` contians train&val&test processes

## Contact
- yanshen0210@gmail.com
* (We will further update relevent information if the paper is accepted!)
