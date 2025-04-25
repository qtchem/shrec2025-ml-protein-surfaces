import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

__all__ = ["generate_features"]

def generate_features(
        features_list,
        training_indices,
        alpha,
        spectra_path,
        pca_numb=None,
        do_pca_at_end=False,
        normalize_features=True,
        labels=None
):
    # 1.npy is PCA_COMp = 100 and The commented out features bellow
    # 2.npy is PCA = 10 and all spectra
    if os.path.exists("3.npy"):
        return np.load("3.npy")
    features = []

    classes, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    H_y = -np.sum(p * np.log(p))

    # 3) Normalized MI relative to H(Y)
    for i, feature in enumerate(features_list):
        print("Feature ", feature)
        load_spectra = np.load(spectra_path + f"{feature[0]}_{feature[1]}_{alpha}_intensities.npz")
        spectra = load_spectra["spectra"]

        if normalize_features:
            stand = MinMaxScaler().fit(spectra[training_indices, :])
            spectra = stand.transform(spectra)

        if pca_numb is not None:
            pca = PCA(n_components=pca_numb + 10, svd_solver="randomized")
            pca = pca.fit(spectra[training_indices, :])
            print("Top/Bottom 5 - Singular values ", pca.singular_values_[:5], pca.singular_values_[-5:])
            print("Top/Bottom 5 - Explained variance ", pca.explained_variance_[:5], pca.explained_variance_[-5:])
            spectra = pca.transform(spectra)
            print(mutual_info_classif(spectra, labels, discrete_features=False, n_jobs=-1) / H_y)

        features.append(spectra)

    features = np.array(features, dtype=float)
    features = np.transpose(features, (1, 0, 2))
    print("Shape of Features ", features.shape)

    if do_pca_at_end:
        print("Perform PCA at the End")
        pca = PCA(n_components=len(features_list) * pca_numb, svd_solver="randomized")
        s = features.shape
        features = features.reshape((s[0], s[1] * s[2]))
        pca.fit(features[training_indices, :])
        print("Top/Bottom 5 - Singular values ", pca.singular_values_[:5], pca.singular_values_[-5:])
        print("Top/Bottom 5 - Explained variance ", pca.explained_variance_[:5], pca.explained_variance_[-5:])
        features = pca.transform(features)
        features = features.reshape((s[0], s[1], pca_numb))
    features = features.astype(np.float32)

    np.save("3.npy", features)
    return features