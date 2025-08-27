# SHREC 2025 Competition: Using a Siamese Neural Network for Deep Metric Learning and Classification of Protein Surfaces
Deep metric learning via triplet loss Siamese neural network and feed-forward classifier for 
SHREC 2025 protein shape classification competition https://shrec2025.drugdesign.fr/.

## Generate Data-Files

The features/inputs to the machine learning models can be reached here: [10.6084/m9.figshare.29995996](https://figshare.com/articles/dataset/SHREC_2025_-_ML_Protein_Surfaces_-_Dr_Heidar-Zadeh_group/29995996).

Please download all of the .npz files and place them in the `./data/features/` folder.

## Installation

To install the dependencies, please run:
```bash
pip install -r requirements.txt
```


## Run Training
To run the training, please modify `configs/siamese.yaml` and run:
```bash
export PYTHONPATH=.
python ./src/train/train_siamese.py ./configs/siamese.yaml
```
