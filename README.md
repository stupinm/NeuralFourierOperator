# Fourier Neural Operator

This is an unofficial PyTorch implementation of the [Fourier Neural Operator for Parametric
Partial Differential Equations
](https://arxiv.org/pdf/2010.08895.pdf) paper by Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar.


## Requirements

The repository is built on top of PyTorch `1.7.1`. 
To install all required packages, run the following command:

```bash
pip install -r requirements.txt
```

## How to use

The **Fourier Neural Operator** (FNO) is an special NN architecture which aimed to solve partial differential equations. 

### Training

To start the training process run from src directory

```bash
python main.py ../cofigs/config_name.json
```

To start a `TensorBoard` simply run

```bash
tensorboard --logdir experiments/experiment_name
```

### Parameters


If you would to modify models' hyperparameters, number of epochs, batch size, and other, 
consider looking to `configs` folder for configuration examples.
An example of the config file is as follows:

```json
command: train&test / train / test / predict 
exp_name: my_experiment                 
experiments: ../experiments             # folder which contain all experiments
dataset: ns_V1e-3_N1200_T50             # name of folder with dataset in appropriate format
datasets: ../datasets                   # folder which contain all datasets
predictions_path: predict               # "predict" command will put predictions into this folder in predictions folder
net_arch: 3d / 2d / 2d_spatial          # 3d architecture utilize spectral convolutions also with time axis, 2d_spatial use default 2d convolutions
pad_coordinates: true                   # whether concatenate spatial coordinates on time axis or not
n_layers: 4                             # number of fourier layers
n_modes_1: 4                            # number of modes to use on -3-axis
n_modes_2: 4                            # number of modes to use on -2-axis
n_modes_3: 4                            # number of modes to use on -1-axis
width: 20                               # w parameter of fourier layer / number of channels
predictive_mode: multiple_step / one_step             # multiple_step predict t_out steps by one forward pass, one_step first predict next timestep then concatenate previous input steps with prediction repeating procedure t_out times; for more details see src/train.py
s: 1                                    # spatial resolution sampling frequency, can be usefull for test/prediction commands
t: 1                                    # time resolution sampling frequency, can be usefull for test/prediction commands
S: 64                                   # spatial size of input and output after downsampling with "s" frequency
t_in: 10                                # number of input timesteps
t_out: 40                               # number of output timesteps
test_ratio: 0.2                         # n_test / n_samples
val_ratio: 0.1                          # n_validation / (n_samples - n_test) 
learning_rate: 0.0025                   # params of Adam optimizer
weight_decay: 0.0001
scheduler_step: 100                     # params of StepLr scheduler
scheduler_gamma: 0.5
kernel_size: 3                          # params for 2d_spatial architecture
padding: 1
seed: 42                                # used to generate train/validation/test split
device: "cuda" / "cpu"


```

There is prepared config in `configs` folder.


This command will create multiple folders

```
experiments/experiment_name
├── models.pth                              # Saved checkpoint
│   
├── tesnorboard                             # Tensorboard event files
│   └── events.out.tfevents
│
│── log.txt                                 # Logs
│
└── config.json                             # Copy of config with added default parameters
```

### Datasets

The following datasets are available: [Navier-Stokes-V1](https://drive.google.com/drive/folders/1KIU2v3GDFwey4tLVyEJW6Q4ImMwKIFxR?usp=sharing). 
If can generate your own dataset using data_generation/main.py script, please look at `data_generation/navier_stokes.py for an example.


## Results

### CelebA

This result is achieved by using the `glf/options/train/train_glf_original_celeba.yml` config.


<p align="center"><img src="imgs/celeba_example.jpg" width="480"\></p>


### Noise interpolation

To play with noise interpolation and image generation check the [notebooks/celeba_demo.ipynb](https://github.com/rakhimovv/GenerativeLatentFlow/blob/master/notebooks/celeba_demo.ipynb).

<p align="center"><img src="imgs/noise_example.png" width="520"\></p>

## Citations

```
@misc{li2020fourier,
      title={Fourier Neural Operator for Parametric Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2010.08895},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{li2020neural,
      title={Neural Operator: Graph Kernel Network for Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2003.03485},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
