import os

import matplotlib.pyplot as plt
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
            roc_score_per_label = metrics.roc_auc_score(y_true=y_true[:,i], y_score=y_prob_pred[:,i])
            auc_roc_values.append(roc_score_per_label)
            fpr, tpr, _ = metrics.roc_curve(y_true=y_true[:,i],  y_score=y_prob_pred[:,i])
        
            axs.plot([0,1], [0,1], 'k--')
            axs.plot(fpr, tpr, 
                label=f'{class_name_list[i]} - AUC = {round(roc_score_per_label, 3)}')

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