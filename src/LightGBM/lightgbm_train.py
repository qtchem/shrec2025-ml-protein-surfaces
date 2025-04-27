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


def dimension_reduction_features(
    feature_list,
    feat_path,
    train_idx,
    alpha,
    pca_comps=50,
    normalize_features=False,
    do_pca=False,
    do_pca_whole=False,
    reshape_features=True,
):
    """Preprocess features with PCA and normalization."""
    print("Total number of feat: ", len(feature_list))
    features = []

    scaler_list = []
    pca_list = []

    for _, feature in enumerate(feature_list):
        print("Feature ", feature)
        load_feat = np.load(feat_path + f"{feature[0]}_{feature[1]}_{alpha}_intensities.npz")
        feat = load_feat["feat"]

        if normalize_features:
            scaler = StandardScaler().fit(feat[train_idx, :])
            feat = scaler.transform(feat)
            scaler_list.append(scaler)
        if do_pca:
            pca = PCA(n_components=pca_comps + 10, svd_solver="randomized")
            pca = pca.fit(feat[train_idx, :])
            print(
                "Top/Bottom 5 - Singular values ",
                pca.singular_values_[:5],
                pca.singular_values_[-5:],
            )
            print(
                "Top/Bottom 5 - Explained variance ",
                pca.explained_variance_[:5],
                pca.explained_variance_[-5:],
            )
            print(f"explained_variance_ratio: {np.sum(pca.explained_variance_ratio_)}")
            # feat = pca.transform(feat)
            feat = pca.transform(feat)
            pca_list.append(pca)

        features.append(feat)

    features = np.array(features, dtype=float)
    features = np.transpose(features, (1, 0, 2))
    print(f"current shape of features: {features.shape}")

    features = features.astype(np.float32)
    if reshape_features:
        features = features.reshape(features.shape[0], -1)

    if do_pca_whole:
        pca = PCA(n_components=pca_comps, svd_solver="randomized")
        pca = pca.fit(features[train_idx, :])
        print(
            "Top/Bottom 5 - Singular values ",
            pca.singular_values_[:5],
            pca.singular_values_[-5:],
        )
        print(
            "Top/Bottom 5 - Explained variance ",
            pca.explained_variance_[:5],
            pca.explained_variance_[-5:],
        )
        print(f"explained_variance_ratio: {np.sum(pca.explained_variance_ratio_)}")
        features = pca.transform(features)

    print(f"final shape of features: {features.shape}")

    if normalize_features and do_pca:
        return features, scaler_list, pca_list
    elif normalize_features and not do_pca:
        return features, scaler_list
    else:
        return features


def train_lightgbm(train_data_path, feat_path):

    df = pd.read_csv(
        train_data_path,
        sep=",",
        header=0,
    )
    # # filter out the classes that are less than 10
    # df_sub = df[df["class_id"].isin(df["class_id"].value_counts()[df["class_id"].value_counts() >= 2].index)]
    # df_not_included = df[~df["class_id"].isin(df_sub["class_id"])]

    # split the data in df_not_included into 2 parts, using stratified sampling

    target_y = df["class_id"]

    # Define the number of splits (folds)
    k = 10
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    indices = np.arange(len(df))

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices, target_y)):

        features, scaler_list, pca_list = dimension_reduction_features(
            feature_list=FEATURE_LIST,
            feat_path=feat_path,
            train_idx=train_idx,
            alpha=100,
            pca_comps=30,
            normalize_features=True,
            do_pca=True,
            reshape_features=True,
            do_pca_whole=False,
        )
        x_train = features[train_idx, :]
        x_val = features[val_idx, :]
        y_train = target_y[train_idx]
        y_val = target_y[val_idx]
        # build up the dataset
        dtrain = lgb.Dataset(x_train, label=y_train)
        dval = lgb.Dataset(x_val, label=y_val)

        # fine tune the model
        params = {
            "objective": "multiclass",
            "num_class": len(df["class_id"].unique()),
            "metric": "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
        }
        # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            callbacks=[early_stopping(100), log_evaluation(100)],
            num_boost_round=100,
        )

        # predictions = np.rint(model.predict(x_val, num_iteration=model.best_iteration))
        pred_val = model.predict(x_val, num_iteration=model.best_iteration)
        pred_val = np.argmax(pred_val, axis=1)
        accuracy = accuracy_score(y_val, pred_val)
        print("Accuracy on validation set: ", accuracy)

        # save the model
        joblib.dump(model, f"lightgbm_model_fold_{fold}.pkl")
        # save the scaler objects
        joblib.dump(scaler_list, f"scaler_fold_{fold}.pkl")
        # save the pca objects
        joblib.dump(pca_list, f"pca_fold_{fold}.pkl")


# best params:
#   Accuracy = 0.6335135135135135
#   Params:
#     objective: multiclass
#     num_class: 97
#     metric: multi_logloss
#     verbosity: -1
#     boosting_type: gbdt
#     feature_pre_filter: False
#     lambda_l1: 1.0383636065216766e-08
#     lambda_l2: 3.4688978486608857
#     num_leaves: 2
#     feature_fraction: 0.748
#     bagging_fraction: 1.0
#     bagging_freq: 0
#     min_child_samples: 10
#     num_iterations: 1000


if __name__ == "__main__":
    # Set the path to the data
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="Path to the training data CSV file",
    )
    parser.add_argument(
        "--feat_path",
        type=str,
        help="Path to the features directory",
    )
    args = parser.parse_args()

    # call the function to train the model
    train_lightgbm(args.train_data_path, args.feat_path)
