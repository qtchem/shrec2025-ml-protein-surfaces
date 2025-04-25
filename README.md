# SHREC 2025 Competition: Using a Siamese Neural Network for Deep Metric Learning and Classification of Protein Surfaces
Deep metric learning via triplet loss Siamese neural network and feed-forward classifier for 
SHREC 2025 protein shape classification competition https://shrec2025.drugdesign.fr/.

## Generate Data-Files

Please contact us or create an issue, if you want the data files or how to generate them.

## Run Training
To run the training, please modify `configs/siamese.yaml` and run:
```bash
export PYTHONPATH=.
python ./src/train/train_siamese.py ./configs/siamese.yaml
```
