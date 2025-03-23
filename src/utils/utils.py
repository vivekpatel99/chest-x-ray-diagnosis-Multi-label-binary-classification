import os

import matplotlib.pyplot as plt
import numpy as np
import opendatasets as od
from sklearn import metrics
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def setup_evnironment_vars()-> None:
    # Set environment variables
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 



def plot_auc_curve(output_dir, class_name_list, y_true, y_prob_pred):
    """Plots the ROC curve for each class and calculates the AUC."""
    n_classes = len(class_name_list)

    # Binarize the true labels (important for multi-label)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_pred[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC - {class_name_list[i]} (area = {roc_auc:0.2f})')

    # Plot the diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2)

    # Set plot limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")

    # Save the figure
    fig.savefig(output_dir / 'ROC-Curve.png')
    plt.close(fig)
    return fig

def download_dataset(log, dataset_path):
    dataset_path.mkdir(parents=True, exist_ok=True)
    if not dataset_path.is_dir():
        log.info("Downloading the dataset")
        dataset_url = 'https://www.kaggle.com/datasets/nih-chest-xrays/sample'
        od.download(dataset_url)