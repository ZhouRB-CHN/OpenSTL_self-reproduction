## Introduction

OpenSTL is a comprehensive benchmark for spatio-temporal predictive learning, encompassing a broad spectrum of methods and diverse tasks, ranging from synthetic moving object trajectories to real-world scenarios such as human motion, driving scenes, traffic flow, and weather forecasting. OpenSTL offers a modular and extensible framework, excelling in user-friendliness, organization, and comprehensiveness. The codebase is organized into three abstracted layers, namely the core layer, algorithm layer, and user interface layer, arranged from the bottom to the top. We support PyTorch Lightning implementation [OpenSTL-Lightning](https://github.com/chengtan9907/OpenSTL/tree/OpenSTL-Lightning) (recommended) and naive PyTorch version [OpenSTL](https://github.com/chengtan9907/OpenSTL/tree/OpenSTL).

<p align="center" width="100%">
  <img src='https://github.com/chengtan9907/OpenSTL/assets/34480960/4f466441-a78a-405c-beb6-00a37e3d3827' width="90%">
</p>

<!-- <p align="center" width="100%">
  <img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/246222226-61e6b8e8-959c-4bb3-a1cd-c994b423de3f.png' width="90%">
</p> -->

## Overview

<details open>
<summary>Major Features</summary>


- **Flexiable Code Design.**
  OpenSTL decomposes STL algorithms into `methods` (training and prediction), `models` (network architectures), and `modules`, while providing unified experiment API. Users can develop their own STL algorithms with flexible training strategies and networks for different STL tasks.

- **Standard Benchmarks.**
  OpenSTL will support standard benchmarks of STL algorithms image with training and evaluation as many open-source projects (e.g., [MMDetection](https://github.com/open-mmlab/mmdetection) and [USB](https://github.com/microsoft/Semi-supervised-learning)). We are working on training benchmarks and will update results synchronizingly.

</details>

<details open>
<summary>Code Structures</summary>


- `openstl/api` contains an experiment runner.
- `openstl/core` contains core training plugins and metrics.
- `openstl/datasets` contains datasets and dataloaders.
- `openstl/methods/` contains training methods for various video prediction methods.
- `openstl/models/` contains the main network architectures of various video prediction methods.
- `openstl/modules/` contains network modules and layers.
- `tools/` contains the executable python files `tools/train.py` and `tools/test.py` with possible arguments for training, validating, and testing pipelines.

</details>



## Installation

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:

```shell
git clone https://github.com/chengtan9907/OpenSTL
cd OpenSTL
conda env create -f environment.yml
conda activate OpenSTL
python setup.py develop
```

<details close>
<summary>Dependencies</summary>


* argparse
* dask
* decord
* fvcore
* hickle
* lpips
* matplotlib
* netcdf4
* numpy
* opencv-python
* packaging
* pandas
* python<=3.10.8
* scikit-image
* scikit-learn
* torch
* timm
* tqdm
* xarray==0.19.0
  </details>

Please refer to [install.md](docs/install.md) for more detailed instructions.

## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage. Here is an example of single GPU non-distributed training SimVP+gSTA on Moving MNIST dataset.

```shell
bash tools/prepare_data/download_mmnist.sh
python tools/train.py -d mmnist --lr 1e-3 -c configs/mmnist/simvp/SimVP_gSTA.py --ex_name mmnist_simvp_gsta
```

## Tutorial on using Custom Data

For the convenience of users, we provide a tutorial on how to train, evaluate, and visualize with OpenSTL on custom data. This tutorial enables users to quickly build their own projects using OpenSTL. For more details, please refer to the [`tutorial.ipynb`](examples/tutorial.ipynb) in the `examples/` directory.

A Colab demo of this tutorial is also provided:

<a href="https://colab.research.google.com/drive/19uShc-1uCcySrjrRP3peXf2RUNVzCjHh?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Visualization

We present visualization examples of ConvLSTM below. For more detailed information, please refer to the [visualization](docs/visualization/).

- For synthetic moving object trajectory prediction and real-world video prediction, visualization examples of other approaches can be found in [visualization/video_visualization.md](docs/visualization/video_visualization.md). BAIR and Kinetics are not benchmarked and only for illustration.

- For traffic flow prediction, visualization examples of other approaches are shown in [visualization/traffic_visualization.md](docs/visualization/traffic_visualization.md).

- For weather forecasting, visualization examples of other approaches are shown in [visualization/weather_visualization.md](docs/visualization/weather_visualization.md).

<div align="center">


|                         Moving MNIST                         |                        Moving FMNIST                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_ConvLSTM.gif' height="auto" width="260" ></div> |

|                      Moving MNIST-CIFAR                      |                         KittiCaltech                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_ConvLSTM.gif' height="auto" width="260" ></div> |

|                             KTH                              |                          Human 3.6M                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_ConvLSTM.gif' height="auto" width="260" ></div> |

|                      Traffic - in flow                       |                      Traffic - out flow                      |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-traffic/taxibj_in_flow_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-traffic/taxibj_out_flow_ConvLSTM.gif' height="auto" width="260" ></div> |

|                    Weather - Temperature                     |                      Weather - Humidity                      |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_temperature_5_625_ConvLSTM.gif' height="auto" width="360" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_humidity_5_625_ConvLSTM.gif' height="auto" width="360" ></div> |

|                   Weather - Latitude Wind                    |                    Weather - Cloud Cover                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_wind_latitude_ConvLSTM.gif' height="auto" width="360" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-weather-5-625/weather_cloud_cover_5_625_ConvLSTM.gif' height="auto" width="360" ></div> |

|                      BAIR Robot Pushing                      |                         Kinetics-400                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <div align=center><img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/257872182-4f31928d-2ebc-4407-b2d4-1fe4a8da5837.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/257872560-00775edf-5773-478c-8836-f7aec461e209.gif' height="auto" width="260" ></div> |

</div>

## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.

