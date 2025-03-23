import os

import matplotlib.pyplot as plt
import opendatasets as od
from sklearn import metrics


def setup_evnironment_vars()-> None:
    # Set environment variables
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 



def plot_auc_curve(output_dir, class_name_list, y_true, y_prob_pred):
    auc_roc_values = []
    fig, axs = plt.subplots(1)
    for i in range(len(class_name_list)):
        try:
            y_true_per_label= y_true[:,i]
            y_pred_per_label= y_prob_pred[:,i]
            roc_score_per_label = metrics.roc_auc_score(y_true=y_true_per_label, y_score=y_pred_per_label)
            auc_roc_values.append(roc_score_per_label)
            false_positive_rates, true_positive_rates, _= metrics.roc_curve(y_true=y_true_per_label,  y_score=y_pred_per_label)
            
            # plt.figure(1, figsize=(13, 13))
            axs.plot([0,1], [0,1], 'k--')
            axs.plot(false_positive_rates, true_positive_rates, 
                label=f'{class_name_list[i]} - AUC = {round(roc_score_per_label, 4)}')

            axs.set_xlabel('False Positive Rate')
            axs.set_ylabel('True Positive Rate')
            axs.legend(loc='lower right')
        except Exception as e:
            print(
            f"Error in generating ROC curve for {class_name_list[i]}. "
            f"Dataset lacks enough examples."
            f"{e}"
        )
    plt.savefig(f"{output_dir}/ROC-Curve.png")

    return fig

def download_dataset(log, dataset_path):
    dataset_path.mkdir(parents=True, exist_ok=True)
    if not dataset_path.is_dir():
        log.info("Downloading the dataset")
        dataset_url = 'https://www.kaggle.com/datasets/nih-chest-xrays/sample'
        od.download(dataset_url)