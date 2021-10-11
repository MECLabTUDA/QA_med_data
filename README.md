# Quality Assurance for Medical Data
Using this repository, six artifact classifiers can be trained, evaluated and used for inference to assure the quality of CT scans. The cassifiers capture the six most commonly known artifacts present in CT scans: blurring, gaussian noise, ghosting, low resolution, motion and spike artifacts. Further, a metric called Lung Fully Captured (LFC) is provided which indicates if the lung is fully captured in a CT scan or not (based on coronal and axial view). For this metric, the [adapted implementation](https://github.com/amrane99/lungmask) from the [original lungmask implementation](https://github.com/JoHof/lungmask) is used, whereas the pre-trained segmentation networks from the 18th of February 2021 -- *first commit after forking original lungmask implementation* -- are used and have not been retrained nor touched/adapted in any way or form. The publication of the original lungmask implementation can be found at:
>Hofmanninger, J., Prayer, F., Pan, J. et al. Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. Eur Radiol Exp 4, 50 (2020). https://doi.org/10.1186/s41747-020-00173-2

Currently, the classifiers and the LFC metric are found under the master branch.

## Installation
The simplest way to install all dependencies is by using [Anaconda](https://conda.io/projects/conda/en/latest/index.html):


1. Create a Python3.8 environment as follows: `conda create -n <your_anaconda_env> python=3.8` and activate the environment.
2. Install CUDA and PyTorch through conda with the command specified by [PyTorch](https://pytorch.org/). The command for Linux was at the time `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`. The code was developed and last tested with the PyTorch version 1.6.
3. Navigate to the project root (where setup.py lives).
4. Execute `pip install -r requirements.txt` to install all required packages.
5. Set your paths in mp.paths.py.
6. Execute `git update-index --assume-unchanged mp/paths.py` so that changes in the paths file are not tracked in the repository.
7. Execute `pytest` to ensure that everything is working. Note that one of the tests will test whether at least one GPU is present, if you do not wish to test this mark to ignore. The same holds for tests that used datasets that have to be previously downloaded.


## Training Evaluation or Inference
To train, evaluate or use the presented artifact classifiers from this repository, please refer to the corresponding [documentation](documentations/JIP.md).
