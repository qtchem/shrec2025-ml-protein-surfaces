import os

import joblib
import numpy as np
import optuna.integration.lightgbm as lgb
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# feature list
FEATURE_LIST = [
    ["electrostatic", "electrostatic"],
    ["electrostatic_neg", "electrostatic_neg"],
    ["electrostatic_pos", "electrostatic_pos"],
    ["gaussian_curvature", "gaussian_curvature"],
    ["principal_curvature", "principal_curvature"],
    ["shape_index", "shape_index"],
    ["radial_distance", "radial_distance"],
    ["electrostatic", "gaussian_curvature"],
    ["electrostatic", "principal_curvature"],
    ["electrostatic", "radial_distance"],
    ["gaussian_curvature", "principal_curvature"],
    ["gaussian_curvature", "shape_index"],
]
FEATURE_PATH = "~/SHREC/train_set_data/"

# load the pca objects
PCA_LIST = joblib.load("pca_fold_0.pkl")
SCALER_LIST = joblib.load("scaler_fold_0.pkl")


def preprocess_test_data(
    feature_list=FEATURE_LIST,
    feat_path="/global/scratch/hpc5605/SHREC/test_set_data/",
    pca_list=PCA_LIST,
    scaler_list=SCALER_LIST,
    alpha=100,
):
    print("Total number of feat: ", len(feature_list))
    features = []

    for pca, scaler, feature in zip(pca_list, scaler_list, feature_list):
        print("Feature ", feature)
        load_feat = np.load(feat_path + f"{feature[0]}_{feature[1]}_{alpha}_intensities.npz")
        feat = load_feat["feat"]
        feat = scaler.transform(feat)
        feat = pca.transform(feat)

        features.append(feat)

    features = np.array(features, dtype=float)
    features = np.transpose(features, (1, 0, 2))

    features = features.astype(np.float32)
    features = features.reshape(features.shape[0], -1)

    return features


test_data = preprocess_test_data(
    feature_list=FEATURE_LIST,
    feat_path=FEATURE_PATH,
    pca_list=PCA_LIST,
    scaler_list=SCALER_LIST,
    alpha=100,
)

# load the model with joblib
model = joblib.load("lightgbm_model_fold_0.pkl")

# run the inference
pred_test = model.predict(test_data, num_iteration=model.best_iteration)
pred_test = np.argmax(pred_test, axis=1)

df = pd.read_csv("test_set_ground_truth.csv", sep=",", header=0)
y_true = df["ground_truth_class"].to_numpy()
accuracy_test = accuracy_score(y_true, pred_test)

print("Accuracy on test set: ", accuracy_test)
# 0.6126669538991814

df["pred_lightgbm"] = pred_test
df.to_csv("test_set_lightgbm_predictions.csv", sep=",", index=False)
