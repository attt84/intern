from __future__ import annotations

import math

import torch


class SegmentationMeter:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)
        self.loss_sum = 0.0
        self.loss_weight = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, *, loss: float | None = None, count: int | None = None) -> None:
        preds = preds.detach().view(-1)
        targets = targets.detach().view(-1)
        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]
        if preds.numel() == 0:
            return

        valid_classes = (targets >= 0) & (targets < self.num_classes)
        preds = preds[valid_classes].to(torch.int64)
        targets = targets[valid_classes].to(torch.int64)
        encoded = (targets * self.num_classes + preds).to("cpu")
        hist = torch.bincount(encoded, minlength=self.num_classes * self.num_classes).reshape(self.num_classes, self.num_classes)
        self.confusion += hist.to(torch.float64)

        if loss is not None and count is not None:
            self.loss_sum += float(loss) * count
            self.loss_weight += int(count)

    def class_iou(self) -> list[float | None]:
        intersection = torch.diag(self.confusion)
        union = self.confusion.sum(dim=1) + self.confusion.sum(dim=0) - intersection
        scores: list[float | None] = []
        for class_index in range(self.num_classes):
            union_value = float(union[class_index].item())
            if union_value <= 0:
                scores.append(None)
            else:
                scores.append(float(intersection[class_index].item() / union_value))
        return scores

    def summary(self) -> dict[str, float | list[float | None]]:
        intersection = torch.diag(self.confusion)
        target_total = self.confusion.sum(dim=1)
        pred_total = self.confusion.sum(dim=0)
        union = target_total + pred_total - intersection

        valid_iou = union > 0
        valid_accuracy = target_total > 0

        iou = torch.zeros_like(intersection)
        accuracy = torch.zeros_like(intersection)
        iou[valid_iou] = intersection[valid_iou] / union[valid_iou]
        accuracy[valid_accuracy] = intersection[valid_accuracy] / target_total[valid_accuracy]

        miou = float(iou[valid_iou].mean().item()) if valid_iou.any() else 0.0
        mean_accuracy = float(accuracy[valid_accuracy].mean().item()) if valid_accuracy.any() else 0.0
        total_pixels = float(target_total.sum().item())
        pixel_accuracy = float(intersection.sum().item() / total_pixels) if total_pixels > 0 else 0.0
        loss = self.loss_sum / self.loss_weight if self.loss_weight > 0 else math.nan

        return {
            "loss": loss,
            "miou": miou,
            "pixel_accuracy": pixel_accuracy,
            "mean_accuracy": mean_accuracy,
            "class_iou": self.class_iou(),
        }
