The code is built on [mdistiller]([https://github.com/megvii-research/mdistiller/releases/tag/checkpoints](https://github.com/megvii-research/mdistiller/tree/master))
# LumiNet: The Bright Side of Perceptual Knowledge Distillation
### Embarrassingly simple knowledge distillation method
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/luminet-the-bright-side-of-perceptual/classification-on-cifar-100)](https://paperswithcode.com/sota/classification-on-cifar-100?p=luminet-the-bright-side-of-perceptual)
## Framework & Performance
![image](https://github.com/ismail31416/LumiNet/assets/87028897/c3c0c715-1daf-4af2-a300-3b6429ede0b4)

### CIFAR-100 Benchmark Results ( Same Architecture ):

| Teacher <br> Student |ResNet56 <br> ResNet20|ResNet110 <br> ResNet32| ResNet32x4 <br> ResNet8x4| WRN-40-2 <br> WRN-16-2| WRN-40-2 <br> WRN-40-1 | VGG13 <br> VGG8|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|
| KD | 70.66 | 73.08 | 73.33 | 74.92 | 73.54 | 72.98 |
| **LumiNet** | **72.29** | **74.20** | **77.50** | **76.38** | **75.12** | **74.94** |

### CIFAR-100 Benchmark Results (Heterogeneous architecture):

| Teacher <br> Student |ResNet32x4 <br> ShuffleNet-V1|WRN-40-2 <br> ShuffleNet-V1| VGG13 <br> MobileNet-V2| ResNet50 <br> MobileNet-V2| ResNet32x4 <br> ShuffleNet-V2|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|
| KD | 74.07 | 74.83 | 67.37 | 67.35 | 74.45 |
| **LumiNet** | **76.66** | **76.95** | **70.50** | **70.97** | **77.55** |

### ImageNet Benchmark Results:
On ImageNet:

| Teacher <br> Student |ResNet34 <br> ResNet18|ResNet50 <br> MobileNet-V1|
|:---------------:|:-----------------:|:-----------------:|
| KD | 71.03 | 70.50 | 
| **LumiNet** | **71.89** | **72.55** |

### Installation

**Supported Environments:**

- Python version 3.6
- PyTorch version 1.9.0
- torchvision version 0.10.0

**Install the package:**

```bash
sudo pip3 install -r requirements.txt
sudo python3 setup.py develop
```
## Getting Started

### 1. Wandb Integration for Logging

- Register on Wandb: [Wandb Registration](https://wandb.ai/home).
- To opt-out of Wandb logging, set `CFG.LOG.WANDB` to `False` in `mdistiller/engine/cfg.py`.

### 2. Evaluation

- Evaluate the performance of provided or custom-trained models.

- Download our models from [this link](https://github.com/) and save the checkpoints in `./download_ckpts`.

- For ImageNet testing, download the dataset from [ImageNet](https://image-net.org/) and place it in `./data/imagenet`.

  ```bash
  # Evaluate teachers
  python3 tools/eval.py -m resnet32x4 # resnet32x4 on cifar100
  python3 tools/eval.py -m ResNet34 -d imagenet # ResNet34 on imagenet
  
  # Evaluate students
  python3 tools/eval.p -m resnet8x4 -c download_ckpts/luminet_resnet8x4 # luminet-resnet8x4 on cifar100
  python3 tools/eval.p -m MobileNetV1 -c download_ckpts/imgnet_luminet_mv1 -d imagenet # luminet-mv1 on imagenet
  python3 tools/eval.p -m model_name -c output/your_exp/student_best # your checkpoints
  ```

### 3. Training on CIFAR-100
- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

  ```bash
  # for instance, our LumiNet method.
  python3 tools/train.py --cfg configs/cifar100/luminet/res32x4_res8x4.yaml

  # you can also change settings at command line
  python3 tools/train.py --cfg configs/cifar100/luminet/res32x4_res8x4.yaml SOLVER.BATCH_SIZE 128 SOLVER.LR 0.1
  ```

### 4. Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`

  ```bash
  # for instance, our LumiNet method.
  python3 tools/train.py --cfg configs/imagenet/r34_r18/luminet.yaml
  ```

### 5. Training on MS-COCO ( This part will be released soon)

- see [detection.md](detection/README.md)


### 6. Extension: Visualizations

- Jupyter notebooks: [tsne](tools/visualizations/tsne.ipynb) and [correlation_matrices](tools/visualizations/correlation.ipynb)

# Citation

If this repo is helpful for your research, please consider citing the paper:

```BibTeX
@article{hossain2023luminet,
  title={LumiNet: The Bright Side of Perceptual Knowledge Distillation},
  author={Hossain, Md Ismail and Elahi, MM and Ramasinghe, Sameera and Cheraghian, Ali and Rahman, Fuad and Mohammed, Nabeel and Rahman, Shafin},
  journal={arXiv preprint arXiv:2310.03669},
  year={2023}
}
```



# Acknowledgement

- We would like to extend our sincere appreciation to the contributors of mdistiller for their dedicated efforts and significant contributions.





