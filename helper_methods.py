import numpy as np
import torch
from lime import lime_image
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch import nn
from typing import List, Tuple, Any


def add_counts(ax: plt.Axes) -> None:
    """
    Annotate a bar plot with counts above each bar.

    Args:
        ax (plt.Axes): The matplotlib Axes object representing the bar plot.
    """
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(
                format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                xytext=(0, 9),
                textcoords='offset points'
            )


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], labels: List[str], title: str = "Confusion Matrix") -> None:
    """
    Plot a confusion matrix with the given true and predicted labels.

    Args:
        y_true (List[int]): List of true labels.
        y_pred (List[int]): List of predicted labels.
        labels (List[str]): List of label names to display on the confusion matrix.
        title (str): Title of the plot. Default is "Confusion Matrix".
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()


def plot_classification_report(y_true: List[int], y_pred: List[int], labels: List[str]) -> None:
    """
    Print the classification report for the given true and predicted labels.

    Args:
        y_true (List[int]): List of true labels.
        y_pred (List[int]): List of predicted labels.
        labels (List[str]): List of label names to display in the report.
    """
    report = classification_report(y_true, y_pred, target_names=labels)
    print("Classification Report:\n")
    print(report)


def explain_age_with_lime_samples(
    model: nn.Module, 
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, str]], 
    device: str = 'cuda'
) -> None:
    """
    Generate LIME explanations for age predictions and display them.

    Args:
        model (nn.Module): The trained model for age prediction.
        samples (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, str]]): List of samples, where each sample contains
            an image tensor, predicted age tensor, true age tensor, additional data (if any), and image name.
        device (str): The device to run the model on ('cuda' or 'cpu'). Default is 'cuda'.
    """
    explainer = lime_image.LimeImageExplainer()
    model.to(device)

    num_samples = len(samples)
    num_per_row = 5
    num_rows = (num_samples + num_per_row - 1) // num_per_row

    fig, axes = plt.subplots(num_rows, num_per_row, figsize=(num_per_row * 4, num_rows * 4))
    axes = axes.flatten()

    for i, (image, age_pred, age_label, _, img_name) in enumerate(samples):
        if i >= len(axes):
            break

        image_np = image.numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        def predict_fn(imgs: np.ndarray) -> np.ndarray:
            model.eval()
            imgs = torch.tensor(imgs).permute(0, 3, 1, 2).float().to(device)
            with torch.no_grad():
                age_preds, _ = model(imgs)
            return age_preds.cpu().numpy()

        explanation = explainer.explain_instance(image_np, predict_fn, top_labels=1, num_samples=1000)

        pred_age = age_pred.item()
        real_age = age_label.item()

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=False)

        darkened_image = image_np * 0.5
        highlighted_image = np.where(mask[:, :, np.newaxis], temp, darkened_image)

        axes[i].imshow(mark_boundaries(highlighted_image, mask))
        axes[i].set_title(f'Real Age: {real_age}, Pred Age: {pred_age:.1f}')
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def explain_gender_with_lime_samples(
    model: nn.Module, 
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, str]], 
    device: str = 'cuda'
) -> None:
    """
    Generate LIME explanations for gender predictions and display them.

    Args:
        model (nn.Module): The trained model for gender prediction.
        samples (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, str]]): List of samples, where each sample contains
            an image tensor, predicted gender tensor, true gender tensor, additional data (if any), and image name.
        device (str): The device to run the model on ('cuda' or 'cpu'). Default is 'cuda'.
    """
    explainer = lime_image.LimeImageExplainer()
    model.to(device)

    num_samples = len(samples)
    num_per_row = 5
    num_rows = (num_samples + num_per_row - 1) // num_per_row

    fig, axes = plt.subplots(num_rows, num_per_row, figsize=(num_per_row * 4, num_rows * 4))
    axes = axes.flatten()

    for i, (image, gender_pred, gender_label, _, img_name) in enumerate(samples):
        if i >= len(axes):
            break

        image_np = image.numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        def predict_fn(imgs: np.ndarray) -> np.ndarray:
            model.eval()
            imgs = torch.tensor(imgs).permute(0, 3, 1, 2).float().to(device)
            with torch.no_grad():
                _, gender_preds = model(imgs)
            return nn.Softmax(dim=1)(gender_preds).cpu().numpy()

        explanation = explainer.explain_instance(image_np, predict_fn, top_labels=2, num_samples=1000)

        pred_gender = np.argmax(gender_pred)
        real_gender = gender_label.item()

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=False)

        darkened_image = image_np * 0.5
        highlighted_image = np.where(mask[:, :, np.newaxis], temp, darkened_image)

        axes[i].imshow(mark_boundaries(highlighted_image, mask))
        axes[i].set_title(f'Real Gender: {real_gender}, Pred Gender: {pred_gender}')
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
