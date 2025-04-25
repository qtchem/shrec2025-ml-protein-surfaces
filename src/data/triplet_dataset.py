import numpy as np
from torch.utils.data import Dataset

__all__ = ['TripletDataset', 'TripletDatasetWeighted']


class TripletDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

        # Create a mapping from class labels to indices
        self.class_to_indices = {}
        for index, label in enumerate(labels):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(index)

        # List of all unique classes
        self.classes = list(self.class_to_indices.keys())
        print("Unique classes: ", np.sort(self.classes))
        print("Missing ", set(range(0, 97)).difference(set(self.classes)))

    def __len__(self):
        """Return the total number of samples."""
        return len(self.features)

    def __getitem__(self, index):
        """
        Returns a triplet (anchor, positive, negative).
        """
        # Get the anchor and its label
        anchor = self.features[index]
        anchor_label = self.labels[index]

        # Select a positive sample: a different index with the same class
        positive_index = index
        if len(self.class_to_indices[anchor_label]) != 1:
            while positive_index == index:
                positive_index = np.random.choice(self.class_to_indices[anchor_label])
        positive = self.features[positive_index]

        # Select a negative sample: sample from a different class than the anchor
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = np.random.choice(self.classes)
        negative_index = np.random.choice(self.class_to_indices[negative_label])
        negative = self.features[negative_index]

        return anchor, positive, negative


class TripletDatasetWeighted(Dataset):
    r"""
    Triplet Loss Function with Hard-

    """
    def __init__(self, features, labels, difficulty_scores):
        self.features = features
        self.labels = labels
        self.difficulty_scores = difficulty_scores  # Should be a numpy array of scores
        # Normalize scores to get probabilities
        self.probs = self.difficulty_scores / np.sum(self.difficulty_scores)
        print(self.probs, self.probs.shape)
        # Existing mapping from class to indices as before
        self.class_to_indices = {}
        for index, label in enumerate(labels):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(index)

        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return len(self.features)

    def update_probs_2(self, probs):
        self.probs = probs

    def update_probs(self, indices_wrong, w=2):
        K = len(indices_wrong)
        N = len(self.features)
        denom = K* w + (N - K)
        self.probs = np.ones(N) / denom
        for i in indices_wrong:
            self.probs[i] = w /denom

        print("Updated probability", np.sum(self.probs))
        print("Length of wrong ones and Fraction: ", K, K / N)
        print("Individual Probability of wrong ones and any", w/ denom, K * w /denom)
        print("probability of right ones and any", 1 / denom, (N - K) / denom)

    def __getitem__(self, _):
        # Sample an anchor index using the computed probabilities.
        anchor_index = np.random.choice(len(self.features), p=self.probs)
        # Ignore anchors whose class indices is one!, because they have only one positive
        while len(self.class_to_indices[self.labels[anchor_index]]) == 1:
            anchor_index = np.random.choice(len(self.features), p=self.probs)
        anchor = self.features[anchor_index]
        anchor_label = self.labels[anchor_index]

        # Positive: sample another instance of the same class.
        positive_index = anchor_index
        if len(self.class_to_indices[anchor_label]) != 1:
            while positive_index == anchor_index:
                positive_index = np.random.choice(self.class_to_indices[anchor_label])
        positive = self.features[positive_index]

        # Negative: choose a sample from a different class.
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = np.random.choice(self.classes)
        negative_index = np.random.choice(self.class_to_indices[negative_label])
        negative = self.features[negative_index]

        return anchor, positive, negative
