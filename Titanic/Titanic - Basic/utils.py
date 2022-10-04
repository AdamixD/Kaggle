import matplotlib.pyplot as plt
import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def get_new_categorical_attributes(transform_pipeline):
    categories = transform_pipeline.named_transformers_['cat']['cat_encoder'].categories_
    new_cat_attributes = []

    for cat_set in categories:
        for cat in cat_set:
            new_cat_attributes.append(cat)

    return new_cat_attributes


def add_new_feature(X, family_members=True, not_alone=True):
    if family_members and not_alone:
        X["Family Members"] = X["Parch"] + X["SibSp"]
        X["Not Alone"] = (X["Family Members"] != 0)
    elif not family_members and not_alone:
        X.drop(["Family Members"], axis=1)
    elif family_members and not not_alone:
        X["Family Members"] = X["Parch"] + X["SibSp"]
    return X



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.grid(True)
    plt.xlabel("Threshold")
    plt.legend(loc="center right", fontsize=12)
    plt.figure(figsize=(8, 4))
    plt.show()


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    plt.xlabel('Percentage of false positives', fontsize=12)
    plt.ylabel('Percentage of true positives', fontsize=12)
    plt.figure(figsize=(8, 4))
    plt.show()


def plot_roc_curves(fpr_1, tpr_1, fpr_2, tpr_2, fpr_3, tpr_3, label_1=None, label_2=None, label_3=None):
    plt.plot(fpr_1, tpr_1, "b-", linewidth=2, label=label_1)
    plt.plot(fpr_2, tpr_2, "r-", linewidth=2, label=label_2)
    plt.plot(fpr_3, tpr_3, "g-", linewidth=2, label=label_3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    plt.xlabel('Percentage of false positives', fontsize=12)
    plt.ylabel('Percentage of true positives', fontsize=12)
    plt.legend(loc="center right", fontsize=12)
    plt.figure(figsize=(8, 4))
    plt.show()
