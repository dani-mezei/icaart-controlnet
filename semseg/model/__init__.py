import torch
from pytorch_lightning import LightningModule
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights
from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassJaccardIndex


class DeeplabV3Resnet101(LightningModule):
    def __init__(self, num_classes=19, lr_rate=1e-4, class_names=None):
        """
        Initializes the DeeplabV3Resnet101 model.

        Args:
            num_classes (int): Number of classes for the model's output.
            lr_rate (float): Learning rate for the optimizer.
            class_names (list, optional): List of class names for logging per-class metrics.
                                         Defaults to None, logging with index.
        """
        super(DeeplabV3Resnet101, self).__init__()
        self.save_hyperparameters()  # Saves num_classes, lr_rate, class_names

        self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        # --- Metrics Initialization ---
        # Training metric: Mean IoU (mIoU)
        # MulticlassJaccardIndex calculates mean IoU by default (macro average)
        self.train_miou_metric = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=255
        )

        # Validation metric: Mean IoU (mIoU)
        self.val_miou_metric = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=255
        )

        # Validation metric: Per-class IoU
        # Setting average=None returns a tensor with IoU for each class
        self.val_iou_per_class_metric = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=255, average=None
        )

        # Test metric: Mean IoU (mIoU)
        self.test_miou_metric = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=255
        )

        # Test metric: Per-class IoU
        self.test_iou_per_class_metric = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=255, average=None
        )

        # Store class names if provided for better logging
        self.class_names = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "traffic light",
            7: "traffic sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motorcycle",
            18: "bicycle",
        }
        if self.class_names is not None:
            assert (
                len(self.class_names) == num_classes
            ), "Number of class names must match num_classes"

    def forward(self, x):
        return self.model(x)["out"]

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate and log training mIoU
        miou = self.train_miou_metric(outputs, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train_miou", miou, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate validation mIoU and per-class IoU
        miou = self.val_miou_metric(outputs, labels)
        iou_per_class = self.val_iou_per_class_metric(outputs, labels)

        # Log validation loss and mIoU
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "val_miou",
            miou,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        # Log per-class IoU scores
        for i, iou_score in enumerate(iou_per_class):
            class_label = f"class_{i}"
            if self.class_names is not None:
                class_label = self.class_names[i]
            log_key = f"val_iou_{class_label}".replace(" ", "_")
            self.log(
                log_key,
                iou_score,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate test mIoU and per-class IoU
        miou = self.test_miou_metric(outputs, labels)
        iou_per_class = self.test_iou_per_class_metric(outputs, labels)

        # Log test loss and mIoU
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "test_miou",
            miou,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        # Log per-class IoU scores
        for i, iou_score in enumerate(iou_per_class):
            class_label = f"class_{i}"
            if self.class_names is not None:
                class_label = self.class_names[i]
            log_key = f"test_iou_{class_label}".replace(" ", "_")
            self.log(
                log_key,
                iou_score,
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr_rate)
        # You might want to add a learning rate scheduler here as well
        # Example:
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        return optimizer
