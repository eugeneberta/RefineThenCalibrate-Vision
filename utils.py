import torch
import numpy as np
import lightning as L
import torch.nn.functional as F
from probmetrics.metrics import Metrics
from probmetrics.distributions import CategoricalLogits
from probmetrics.calibrators import TemperatureScalingCalibrator

class Cutout(object):
    # From https://github.com/uoguelph-mlrg/Cutout
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

class CalibrationRefinementTSModule(L.LightningModule):
    # Lightning module to benchmark early stopping metrics.
    def __init__(self, model, lr_scheduler='cosine'):
        super().__init__()
        self.model = model
        self.lr_scheduler = lr_scheduler

        # Buffers to store validation data and targets
        self.val_logits = []
        self.val_labels = []
        self.test_logits = []
        self.test_labels = []
        self.logs = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        logits = self.model(x)

        # Saving validation and test logits to buffer for future logging
        if dataloader_idx == 0:
            self.val_logits.append(logits.detach().cpu())
            self.val_labels.append(y.detach().cpu())
        elif dataloader_idx == 1:
            self.test_logits.append(logits.detach().cpu())
            self.test_labels.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        # Consolidate outputs
        val_logits = torch.cat(self.val_logits)
        val_labels = torch.cat(self.val_labels)
        test_logits = torch.cat(self.test_logits)
        test_labels = torch.cat(self.test_labels)

        # Reset buffers
        self.val_logits.clear()
        self.val_labels.clear()
        self.test_logits.clear()
        self.test_labels.clear()

        # Fitting TS on the validation set using probmetrics
        calibrator = TemperatureScalingCalibrator(opt='bisection', max_bisection_steps=30)
        calibrator.fit_torch(CategoricalLogits(val_logits), val_labels)
        invtemp = calibrator.invtemp_ # Re-scaling parameter

        metrics = Metrics.from_names(['logloss', 'brier', 'ece-15', 'smece']) # Metrics to compute before and after TS.
        invariant_metrics = Metrics.from_names(['class-error', 'auroc-ovr']) # Metrics invariant by TS, treated separately to avoid computing them twice.

        # Compute val metrics
        val_metrics = metrics.compute_all_from_labels_logits(val_labels, val_logits)
        val_metrics.update(invariant_metrics.compute_all_from_labels_logits(val_labels, val_logits))
        val_ts_metrics = metrics.compute_all_from_labels_logits(val_labels, invtemp*val_logits)

        # Compute test metrics
        test_metrics = metrics.compute_all_from_labels_logits(test_labels, test_logits)
        test_metrics.update(invariant_metrics.compute_all_from_labels_logits(test_labels, test_logits))
        test_ts_metrics = metrics.compute_all_from_labels_logits(test_labels, invtemp*test_logits)

        # Logging, _ts stands for "after temperature scaling"
        results = {
            'logloss/val': val_metrics['logloss'].item(),
            'logloss/val_ts': val_ts_metrics['logloss'].item(),
            'logloss/val_gap': val_metrics['logloss'].item() - val_ts_metrics['logloss'].item(),
            'logloss/test': test_metrics['logloss'].item(),
            'logloss/test_ts': test_ts_metrics['logloss'].item(),
            'logloss/test_gap': test_metrics['logloss'].item() - test_ts_metrics['logloss'].item(),
            'brier/val': val_metrics['brier'].item(),
            'brier/val_ts': val_ts_metrics['brier'].item(),
            'brier/val_gap': val_metrics['brier'].item() - val_ts_metrics['brier'].item(),
            'brier/test': test_metrics['brier'].item(),
            'brier/test_ts': test_ts_metrics['brier'].item(),
            'brier/test_gap': test_metrics['brier'].item() - test_ts_metrics['brier'].item(),
            'error/val_error': val_metrics['class-error'].item(),
            'error/val_auroc': val_metrics['auroc-ovr'].item(),
            'error/test_error': test_metrics['class-error'].item(),
            'error/test_auroc': test_metrics['auroc-ovr'].item(),
            'ece/val': val_metrics['ece-15'].item(),
            'ece/val_ts': val_ts_metrics['ece-15'].item(),
            'ece/test': test_metrics['ece-15'].item(),
            'ece/test_ts': test_ts_metrics['ece-15'].item(),
            'smece/val': val_metrics['smece'].item(),
            'smece/val_ts': val_ts_metrics['smece'].item(),
            'smece/test': test_metrics['smece'].item(),
            'smece/test_ts': test_ts_metrics['smece'].item()
        }
        self.log_dict(results, on_step=False, on_epoch=True)
        self.logs.append(results)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

        if self.lr_scheduler=='cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        elif self.lr_scheduler=='step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        else:
            raise Exception('Oops, requested scheduler does not exist!')

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
