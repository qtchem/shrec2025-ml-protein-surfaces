import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import yaml

from src.features import generate_features
from src.data.triplet_dataset import TripletDatasetWeighted
from src.data.classification_dataset import ClassificationDataset
from src.models.siamese import SiameseResidualNetwork
from src.models.mlp import MLP_NN, MLP_RESIDUAL_NN


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main(*args):
    cfg = load_config(args[0])

    # 1) Load your CSV and targets
    df = pd.read_csv(cfg["data"]["train_csv"])
    y = df["class_id"].to_numpy()
    indices = np.arange(len(y))
    n_classes = len(np.unique(y))

    # Probabilities of the hard-example
    probabilites = np.ones(len(indices)) / len(indices)
    if cfg["loss"]["weight_class"]:
        freq_series = df["class_id"].value_counts().sort_index()
        frequency_list = freq_series.values.tolist()
        print("Frequency of each class: ", frequency_list)
        frequency_list = np.array(frequency_list, dtype=np.float32)
        weights = 1.0 / frequency_list
        weights = weights / np.sum(weights)
        print(weights)
        probabilites = np.array([weights[y[i]] / frequency_list[y[i]] for i in indices])

    # 2) Cross‐validation split
    kf = StratifiedKFold(n_splits=cfg["training"]["folds"], shuffle=True, random_state=cfg["random_seed"])
    best_loss = float("inf")

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices, y)):
        print(f"\n=== Fold {fold+1} ===")

        # 3) Generate / cache features
        features = generate_features(
            cfg["features_list"],
            train_idx,
            alpha=cfg["alpha"],
            spectra_path=cfg["data"]["spectra_path"],
            pca_numb=cfg["pca_components"],
            normalize_features=cfg["normalize_features"],
            labels=y
        )

        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 4) Build weighted triplet dataset + loader
        train_ds = TripletDatasetWeighted(X_train, y_train, difficulty_scores=probabilites[train_idx])#np.ones(len(X_train)))
        train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)

        # 5) Model, loss, optimizer_siam, scheduler_siam
        expr_context = {
            "len": len,
            "features_list": cfg["features_list"],
        }
        cfg_siamese = cfg["model"]["siamese"]
        emb_size = eval(cfg_siamese["embedding_size"], {}, expr_context)
        layers = [eval(l, {}, expr_context) for l in cfg_siamese["layers"]]
        model = SiameseResidualNetwork(
            num_spectra=len(cfg["features_list"]),
            layers=layers,
            embedding_size=emb_size,
            dropout_prob=cfg_siamese["dropout_prob"],
            len_spectrum=X_train.shape[2],
        ).to(cfg["device"])

        # Set up classifier
        # Create a separate dataset and loader for validation classification metrics
        class_training_dataset = ClassificationDataset(X_train, y_train)
        train_loader_cls = DataLoader(class_training_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
        class_val_dataset = ClassificationDataset(X_val, y_val)
        val_loader_cls = DataLoader(class_val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
        model_classifier = MLP_NN(
            embedding = emb_size,
            layers=cfg["model"]["classifier"]["layers"],
            num_classes=n_classes,
            dropout_prob=cfg["model"]["classifier"]["dropout_prob"],
        )
        # model_classifier = MLP_RESIDUAL_NN(
        #     embedding = emb_size,
        #     layers_residual=[emb_size],
        #     layers=cfg["model"]["classifier"]["layers"],
        #     num_classes=n_classes,
        #     dropout_prob=cfg["model"]["classifier"]["dropout_prob"],
        # )
        model_classifier.to(cfg["device"])

        # Siamese Training Information
        cfg_optimizer = cfg["training"]["optimizer_siamese"]
        optimizer_siam = optim.Adam(model.parameters(), lr=cfg_optimizer["lr"], weight_decay=cfg_optimizer["weight_decay"])
        scheduler_siam = torch.optim.lr_scheduler.StepLR(optimizer_siam, step_size=cfg_optimizer["scheduler_stepsize"], gamma=0.5)
        triplet_loss = nn.TripletMarginLoss(margin=cfg["loss"]["margin"], p=2)

        # Classifier Training Information
        cfg_optimizer = cfg["training"]["optimizer_classifier"]

        # Define loss function and optimizer.
        if cfg["loss"]["weight_class"]:
            print("Use class weights")              # Imbalanced dataset, weight the class
            weights_tensor = torch.FloatTensor(weights).to(cfg["device"])
            classification_loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            classification_loss_fn = nn.CrossEntropyLoss()

        # 6) Training loop
        hard_indices = []
        for epoch in range(cfg["training"]["epochs"]):
            if hard_indices:
                # Update the hard indices
                train_ds.update_probs(hard_indices, w=cfg["loss"]["hard_weight"])

            # Train the Siamese Neural network
            model.train()
            running_loss, total = 0.0, 0
            for anc, pos, neg in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                anc, pos, neg = anc.to(cfg["device"]), pos.to(cfg["device"]), neg.to(cfg["device"])
                optimizer_siam.zero_grad()
                a_emb = model.forward_once(anc)
                p_emb = model.forward_once(pos)
                n_emb = model.forward_once(neg)
                loss = triplet_loss(a_emb, p_emb, n_emb)
                loss.backward()
                optimizer_siam.step()

                running_loss += loss.item() * anc.size(0)
                total += anc.size(0)

            avg_loss = running_loss / total
            scheduler_siam.step()
            print(f"[Epoch {epoch+1}/{cfg['training']['epochs']}] Triplet Loss: {avg_loss:.4f}  Learning Rate {scheduler_siam.get_last_lr()}")

            # (optional) checkpoint on best_loss



            if epoch % cfg["training"]["epoch_train_classifier"] == 0: # and epoch > 1:
                optimizer_classifier = optim.Adam(model_classifier.parameters(), lr=cfg_optimizer['lr'],
                                                   weight_decay=cfg_optimizer['weight_decay'])
                scheduler_classifier = torch.optim.lr_scheduler.StepLR(optimizer_classifier,
                                                                       step_size=cfg_optimizer["scheduler_stepsize"],
                                                                       gamma=0.5)
                model.eval()  #  freeze Siamese embedding (weights + BN stats)

                for epoch_classifier in range(cfg["training"]['epoch_classifier']):
                    model_classifier.train()
                    classifier_loss = 0.0
                    total, correct = 0, 0

                    hard_indices = set({})
                    predictions_class = set({})

                    for X_batch, y_batch in tqdm(train_loader_cls, desc=f"MLP-Classifier Epoch {epoch + 1}",
                                                 leave=False):
                        X_batch = X_batch.cuda()
                        y_batch = y_batch.cuda()

                        # Forward through Siamese to get embeddings
                        """
                        TRAIN THE EMBEDDING TOO
                        """
                        with torch.no_grad():
                            embeddings = model.forward_once(X_batch)

                        # Forward through MLP
                        optimizer_classifier.zero_grad()
                        outputs = model_classifier(embeddings)
                        loss_cls = classification_loss_fn(outputs, y_batch)
                        loss_cls.backward()
                        optimizer_classifier.step()

                        classifier_loss += loss_cls.item() * X_batch.size(0)

                        # Compute training accuracy (optional)
                        _, predicted = torch.max(outputs, dim=1)
                        correct += (predicted == y_batch).sum().item()
                        total += y_batch.size(0)
                        predictions_class.update(set(predicted.cpu().numpy()))

                        wrong_indices_in_batch = np.where(predicted.cpu() != y_batch.cpu())[0]  # array of indices within the batch

                        hard_indices.update(wrong_indices_in_batch)

                    hard_indices = list(hard_indices)

                    scheduler_classifier.step()
                    classifier_loss /= total
                    train_acc = correct / total
                    print(
                        f"Epoch Classifier {epoch_classifier}    MLP Train Loss: {classifier_loss:.4f}, Train Acc: {train_acc:.4f}  Learning Rate {scheduler_classifier.get_last_lr()}   Number of classes predicted {len(predictions_class)}")



                    # 3) --- Validation (using the MLP classifier) ---
                    model.eval()
                    model_classifier.eval()
                    val_loss, val_correct, val_total = 0.0, 0, 0
                    classes_wrong_predictions = set({})
                    predictions_class = set({})
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader_cls:
                            X_batch = X_batch.cuda()
                            y_batch = y_batch.cuda()
                            # Get embeddings
                            embeddings = model.forward_once(X_batch)
                            # Classify
                            outputs = model_classifier(embeddings)
                            loss_val = classification_loss_fn(outputs, y_batch)
                            val_loss += loss_val.item() * X_batch.size(0)
                            # Accuracy
                            _, predicted = torch.max(outputs, dim=1)
                            val_correct += (predicted == y_batch).sum().item()
                            val_total += y_batch.size(0)

                            predictions_class.update(set(predicted.cpu().numpy()))
                            classes_wrong_predictions.update(
                                y_batch.cpu()[np.where(predicted.cpu() != y_batch.cpu())[0]])

                    classes_wrong_predictions = list(classes_wrong_predictions)
                    val_loss /= val_total
                    val_acc = val_correct / val_total
                    print(
                        f"    Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}   Number of classes predicted {len(predictions_class)}")

                    # -- Saving best model --
                    # If this is the best validation accuracy so far, save weights of both models.
                    # if val_acc > best_val_acc:
                    #     best_val_acc = val_acc
                    #
                    #     # Save the Siamese network weights
                    #     torch.save(model.state_dict(),
                    #                f"best_siamese_fold_{fold}_ep_{epoch}_val_{np.round(best_val_acc, 2)}.pt")
                    #     # Save the classifier weights
                    #     torch.save(model_classifier.state_dict(),
                    #                f"best_classifier_{fold}_ep_{epoch}_val_{np.round(best_val_acc, 2)}.pt")
                    #
                    #     print(f"      >> New best model saved! Fold={fold}, Best Val Acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])