import math
import random
import time
from typing import List, Iterable, Tuple, Optional, Callable, Union, Dict

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import make_grid

from src import console
from src.util.image import signed_to_image, get_images_from_iterable
from src.train import Trainer


class RepresentationClassTrainer(Trainer):
    """
    Expects:
        dataset: (image, target, label, ...) items
            label is optional
        model:
            - forward(image) == target
            - encode(image) or encoder(image) == embedding
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._classification_time = 0.
        self._last_classification_time = 0.

    @torch.no_grad()
    def write_step(self):

        inputs = []
        targets = []
        labels = []
        features = []
        outputs = []
        for batch in self.iter_validation_batches():
            if hasattr(self.model, "encode"):
                feature_batch = self.model.encode(batch[0])
            else:
                feature_batch = self.model.encoder(batch[0])

            inputs.append(batch[0].cpu())
            features.append(feature_batch.cpu())
            outputs.append(self.model(batch[0]).cpu())
            targets.append(batch[1].cpu())
            if len(batch) >= 3:
                labels.append(batch[2].cpu())

        inputs = torch.concat(inputs)
        features = torch.concat(features)
        outputs = torch.concat(outputs)
        targets = torch.concat(targets)
        if labels:
            labels = torch.concat(labels)

        self.log_image("validation_features", signed_to_image(features[:128]))
        self.log_image("validation_outputs", signed_to_image(outputs[:128]))

        self.log_scalar(
            "validation_accuracy_argmax",
            (targets.argmax(dim=-1) == outputs.argmax(dim=-1)).to(torch.float32).mean()
        )
        #print("TARGETS, OUTPUTS:", labels[:10], targets[:10], outputs[:10].round(decimals=3))
        #print("TARGETS, OUTPUTS:", targets[:10].argmax(dim=-1), outputs[:10].argmax(dim=-1))

        errors_per_sample = (outputs - targets).abs().mean(dim=-1)
        best_indices = errors_per_sample.argsort()
        images = [inputs[i] for i in best_indices[:10]]
        images += [inputs[i] for i in reversed(best_indices[-10:])]
        self.log_image("validation_best_worse", make_grid(images, nrow=10))

        self.writer.add_embedding(features[:128], global_step=self.num_input_steps, tag="validation_embedding")

        if len(labels):
            time_since = time.time() - self._last_classification_time
            time_ratio = self._classification_time / max(1., time_since)
            # don't eat more than X% of computation time
            if time_ratio <= 0.07:
                from sklearn.svm import SVC

                classifier = SVC()
                self._last_classification_time = time.time()
                classifier.fit(features, labels)
                self._classification_time = time.time() - self._last_classification_time

                self.log_scalar("validation_accuracy_embedding_svc", classifier.score(features, labels))
