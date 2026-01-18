from supervised_soup.checkpoints import load_best_checkpoint
from supervised_soup.dataloader import get_test_dataloader
from supervised_soup.config import DEVICE
import torch
import wandb
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, roc_auc_score



def evaluate_model(model, log_to_wandb=True):
    """
    Evaluates the (best-model) cehckpoint on the final test set.

    Returns:
    - dict of evaluation metrics including accuracy, F1 macro, top-5, macro AUC-ROC, and per-class accuracy
    """

    device = DEVICE

    # Load the best checkpoint from W&B
    checkpoint = load_best_checkpoint(device=device)
    if checkpoint is None:
        raise RuntimeError(
            "Best checkpoint not found. Make sure wandb.run.name matches the training run."
        )

    # Load model weights and set to eval
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # Load the test dataloader
    test_loader = get_test_dataloader()
    num_classes = len(test_loader.dataset.classes)

    all_preds, all_labels, all_probs = [], [], []

    # Loop over the test set to get predictions
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    # Compute standard metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    top5 = top_k_accuracy_score(all_labels, all_preds, k=5, labels=list(range(num_classes)))

    # Compute macro AUC-ROC
    try:
        # One-hot encode labels
        # If you get a pylint not-callable error here, it should be ignorable
        labels_onehot = torch.nn.functional.one_hot(all_labels, num_classes=num_classes).numpy()
        auc_roc = roc_auc_score(labels_onehot, all_probs.numpy(), average="macro", multi_class="ovr")
    except ValueError:
        auc_roc = None  

    # Per-class acc
    per_class_acc = {}
    for i, class_name in enumerate(test_loader.dataset.classes):
        idxs = (all_labels == i)
        if idxs.sum() > 0:
            per_class_acc[class_name] = accuracy_score(all_labels[idxs], all_preds[idxs])
        else:
            per_class_acc[class_name] = None

    # Metrics dict
    metrics = {
        "accuracy": acc,
        "f1_macro": f1,
        "top5": top5,
        "roc_auc_macro": auc_roc,
        "per_class_acc": per_class_acc
    }

    # Log to W&B
    if log_to_wandb and wandb.run is not None:
        wandb.run.summary["test_accuracy"] = acc
        wandb.run.summary["test_f1_macro"] = f1
        wandb.run.summary["test_top5"] = top5
        wandb.run.summary["test_roc_auc_macro"] = auc_roc
        for cls, acc_cls in per_class_acc.items():
            wandb.run.summary[f"test/{cls}_acc"] = acc_cls

    # Print summary
    print("Test results:")
    print(f"Accuracy: {acc}")
    print(f"F1 Macro: {f1}")
    print(f"Top-5: {top5}")
    print(f"ROC-AUC Macro: {auc_roc}")
    print("Per-class Accuracy:")
    for cls, acc_cls in per_class_acc.items():
        print(f"  {cls}: {acc_cls}")

    return metrics
