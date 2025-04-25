# SHREC 2025 Competition: Using a Siamese Neural Network for Deep Metric Learning and Classification of Protein Surfaces
Deep metric learning via triplet loss Siamese neural network and feed-forward classifier for 
SHREC 2025 protein shape classification competition https://shrec2025.drugdesign.fr/.

To run the training, please modify `configs/siamese.yaml` and run:
```bash
export PYTHONPATH=.
python ./src/train/train_siamese.py ./configs/siamese.yaml
```
