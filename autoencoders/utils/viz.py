import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay


def histogram_error_per_class(error_df, imbalance=False):
    assert (imbalance is False or type(imbalance) == int), "imbalance needs to be either False or an int"
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
    
    class_0 = error_df[error_df['true_class']== 0]
    if imbalance is not False:
        class_0 = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < imbalance)]
    _ = ax[0].hist(class_0.reconstruction_error.values, bins=10)
    ax[0].title.set_text("Normal Transactions")
    
    class1 = error_df[error_df['true_class'] == 1]
    _2 = ax[1].hist(class1.reconstruction_error.values, bins=10)
    ax[1].title.set_text("Fraud Transactions")
    
    for a in ax.flat:
        a.set(xlabel='MSE Reconstruction Error', ylabel='Times Seen')
        
    fig.tight_layout()

def threshold_visualization(error_df, threshold, save_path):
    groups = error_df.groupby('true_class')
    fig, ax = plt.subplots()

    for name, group in groups:
        ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                label= "Fraud" if name == 1 else "Normal")
    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    plt.title("Reconstruction error for different classes")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    plt.show()

def conf_matrix(error_df, threshold, save_path):
    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    matrix = confusion_matrix(error_df.true_class, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show() #save instead

def plot_results(precision, recall, f1, save_path):
    plt.plot(precision.keys(), precision.values(), label="Precision")   
    plt.plot(recall.keys(), recall.values(), label="Recall")
    plt.plot(f1.keys(), f1.values(), label="F1")
    plt.title("Precision, Recall, and F1 Score")
    plt.ylabel("Percent")
    plt.xlabel("MSE Threshold")
    plt.legend()
    plt.show() #save instead