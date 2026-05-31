import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve,
                             precision_recall_curve,
                             confusion_matrix, classification_report,
                             average_precision_score)
import joblib
import os

def plot_roc_curve(y_true, y_pred_proba, 
                   save_path='./data/roc_curve.png'):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    axes[0].plot(fpr, tpr, color='#e74c3c', lw=2,
                 label=f'ROC Curve (AUC = {auc:.4f})')
    axes[0].plot([0,1],[0,1], color='gray', 
                 linestyle='--', label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    axes[1].plot(recall, precision, color='#3498db', lw=2,
                 label=f'PR Curve (AP = {ap:.4f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC+PR curves saved to {save_path}")
    return auc

def plot_confusion_matrix(y_true, y_pred,
                          save_path='./data/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)
    labels = [['TN\n(Correctly Approved)', 'FP\n(Wrongly Rejected)'],
              ['FN\n(Missed Default)', 'TP\n(Correctly Rejected)']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}\n{labels[i][j]}',
                   ha='center', va='center', fontsize=10,
                   color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['Approve','Reject'])
    ax.set_yticklabels(['Non-Default','Default'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")

def get_evaluation_metrics(y_true, y_pred_proba, threshold=0.2):
    y_pred = (y_pred_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = {
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "accuracy": round(report['accuracy'], 4),
        "default_precision": round(report['1']['precision'], 4),
        "default_recall": round(report['1']['recall'], 4),
        "default_f1": round(report['1']['f1-score'], 4)
    }
    print("\n=== MODEL EVALUATION METRICS ===")
    for k, v in metrics.items():
        print(f"{k:25s}: {v}")
    return metrics