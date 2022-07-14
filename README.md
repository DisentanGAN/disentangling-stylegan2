## DisentanGAN: A StyleALAE framework built in PyTorch Lightning

This repository contains a framework for extending the [StyleALAE architecture](https://openaccess.thecvf.com/content_CVPR_2020/html/Pidhorskyi_Adversarial_Latent_Autoencoders_CVPR_2020_paper.html).
Our aim is to provide an easily extensible and comprehensible framework allowing addition of further downstream tasks, such as classification or segmentation.
The greater goal behind that is to challenge the latent space disentanglement problem by incorporating different techniques.

For ease of use, the PyTorch Lightning framework is employed.
This allows for a clear super model containing the model architecture(s) and orchestrating the training in desired manner.

#### Note:

On the **legacy_code** branch, you can find the original codebase, that we forked from.
This codebase differs greatly from ours and might not work with newer versions of PyTorch and other libraries.

Furthermore, we refer to our code as DisentanGAN, even though it bases around the ALAE architecture.
This is merely a style choice.

### Requirements

To install the requirements, execute:

    pip install -r requirements.txt

### Usage

To train, simply execute the training script, e.g.:
    
    python training.py --dataset pcam --run_name example_run_pcam

for a StyleALAE run on the PatchCamelyon (PCAM) dataset or
    
    python training.py --dataset mnist --run_name example_run_mnist --classifier NonLinear --classifier_depth 1 --classifier_classes 10

for a StyleALAE + Classifier on embeddings run on the MNIST dataset.

Further examples can be found in [example_experiments.sh](https://github.com/DisentanGAN/disentangling-stylegan2/blob/master/example_experiments.sh).

#### Default values

We employ a default config to control various hyperparameters.
This config can be found in [defaultvalues.py](https://github.com/DisentanGAN/disentangling-stylegan2/blob/master/defaultvalues.py)
The following default argments are used:

```
default_args = {
    "r1": 10,
    "path_regularize": 2,
    "path_batch_shrink": 2,
    "d_reg_every": 16,
    "g_reg_every": 4,
    "mixing": 0.9,
    "augment": False,
    "augment_p": 0.8,
    "ada_target": 0.6,
    "ada_length": 500000,
    "ada_every": 256,
    "latent": 128,
    "image_size": 32,
    "n_mlp": 8,
    "store_images_every": 1,
    "seed": 42,
    "batch_size": 32,
    "dataloader_workers": 2,
    "classifier": "None",
    "classifier_classes": 10,
    "classifier_depth": 3,
    "checkpoint_path": 'checkpoints/',
    "save_checkpoint_every": 4,
}
```

Each of these is at the same time a command-line argument and can thus be flexibly controlled.
Descriptions for each can be found in the above file.

#### Implementation

The detailed training scheme is orchestrated by the DisentangledSG LightningModule found in [disentangledsg.py](https://github.com/DisentanGAN/disentangling-stylegan2/blob/master/disentangledsg.py).
It contains the Mapping network, Generator, Encoder, and Discriminator as basic structure and, if chosen, the Classifier.

The general structure allows for adding further downstream tasks and their specific optimization schemes.
Each task has their own optimization/regularization functions, prefixed with an underscore (e.g. **_optimize_generation(...)**).
This modularizes the code further.

#### Supported Datasets

Currently, the following datasets are supported and have PyTorch Lightning DataModules implemented.

##### MNIST

The MNIST benchmark dataset is fairly well-known and traditionally used. It consists of 28x28 grayscale images of digits 0 to 9.

##### PatchCamelyon (PCAM)

The [PatchCamelyon](https://github.com/basveeling/pcam) dataset is a newer benchmark dataset derived from the bigger Camelyon16 dataset.
It consists of 327.680 RGB-color images of size 96x96 extracted from histopathologic scans of lymph node sections and contains a positive class (if the 32x32 center pixels contain cancer tissue) and a negative class (if they do not).
For further details, see the link to the original repository.

### Citation

If our code is helpful to you and you use it for a scientific publication, we would appreciate a citation using the following BibTex entry:

```
@misc{disentangan,
    title={{DisentanGAN}: A StyleALAE framework built in PyTorch Lightning},
    author={Biebel, Florian and Hufe, Lorenz and Wiedersich, Niklas Luis and Zimmermann, Sebastian},
    publisher={GitHub},
    howpublished={\url{https://github.com/DisentanGAN/disentangling-stylegan2/}}
}
```

## License

Custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

