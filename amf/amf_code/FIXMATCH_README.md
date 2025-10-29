# FixMatch

This is an unofficial PyTorch implementation of [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685).
The official Tensorflow implementation is [here](https://github.com/google-research/fixmatch).

## Preparing the dataset

Your dataset directory should include `val`,`train` and `test` subdirectories.

## Setting Up the Environment

This section provides instructions on how to set up your development environment to run the project. You have two options: using Docker or setting up a virtual environment.

### Option 1: Using a virtual environment

All the existing requirements for the usual virtual environment should work for Fixmatch - see [here](../../INSTALL.md) for details.

### Option 2: Using Docker

This is legacy, but could be explored if you wanted to containerise your efforts. Follow these steps:

1. **Build the Docker Image**:
   Open a terminal, navigate to the project's root directory (where the Dockerfile is located), and run the following command:

   ```bash
   docker build -f Dockerfile -t fixmatch:latest .
   ```

2. **Run the Docker Image**:

   Specify the path to this directory by replacing `/path/to/local/dataset/folder` in the docker command below with your actual data folder path.

   ```bash
   docker run -it --gpus 'device=0' --shm-size=6g -v $(pwd):/fixmatch -v /path/to/local/dataset/folder:/data fixmatch:latest
   ```

3. **Navigate to the Root Directory**:

   ```bash
   cd ..
   ```

## Running the code

The **base_config.yml** file contains the hyperparameters used on each training run. Prior to running training or testing, the config file parameter `root_path` will need to be updated in order to reflect the path to the directory containing the dataset. If running the code with docker, this will be `/data` otherwise, this is the absolute path to your dataset directory.

The training script can be run from the root folder with:

```bash
python fixmatch/train.py --config fixmatch/config/base_config.yml --out fixmatch/results
```

This will run training using the specified configuration and will output the results to fixmatch/results which will contain a timestamped folder depending on when each experiment is run.

The output folder contains the following files:

- **checkpoint.pth.tar** containing the final version of the model over the training run
- **model_best.pth.tar** containing the best version of the model over the training run
- **config.yaml** containing the configuration used to train the model
- **labelled_data.csv** containing the filenames, column and row information and the class label for each tile in the labelled dataset

The testing script can be run by:

```bash
python fixmatch/evaluate.py --train_folder /path/to/output/training/folder
```

where the path to the training folder is the path to the folder generated during the training stage which contains the model checkpoint and the config file. The test outputs are saved to a test outputs folder inside the training folder.

Outputs include:

- **confusion_matrix.png** - confusion matrix for test data
- **Colonised_incorrect_predictions.png** - incorrect predictions in the 'Colonised' class if any
- **Not Colonised_incorrect_predictions.png** - incorrect predictions in the 'Not Colonised' class if any
- **file_metrics.csv** - per-file metrics
- **test_metrics.csv** - overall metrics on the test data

### Config Parameters

There are a range of config parameters that can be changed in experiments. So far, the majority have been left unchanged and only quantities of training data are changed. The key config parameters are:

- `num_labelled` - number of tile to label from the training dataset
- `arch` - model architectrue, only `'wideresnet'` currently works
- `mu` - the number of unsupervised iters to do per supervised iter
- `lambda_mu` - the weight to apply to the unsupervised loss
- `prop_unlabelled` - this is the proportion of the data left over after labelling to use as unlabelled data (e.g if you have 100 datapoints and set `num_labelled` to 0.2 and `prop_unlabelled` to 0.2 then you would have 20 labelled examples and 16 unlabelled examples)

## References

- [Official TensorFlow implementation of FixMatch](https://github.com/google-research/fixmatch)
- [Unofficial PyTorch implementation of MixMatch](https://github.com/YU1ut/MixMatch-pytorch)
- [Unofficial PyTorch Reimplementation of RandAugment](https://github.com/ildoonet/pytorch-randaugment)
- [PyTorch image models](https://github.com/rwightman/pytorch-image-models)

## Citations

```
@article{sohn2020fixmatch,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:2001.07685},
    year={2020},
}
```
