import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from src.train import Trainer
from src.util.image import make_grid_labeled


class ClassifierTrainer(Trainer):

    def __init__(
            self,
            num_classes: int,
            off_value: float = 0.,
            on_value: float = 1.,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_classes = num_classes
        self._off_value = off_value
        self._on_value = on_value
        self._class_logits_map = dict()

    def train_step(self, input_batch):
        input, labels = input_batch
        expected_logits = torch.concat([
            self.get_logits(label).unsqueeze(0)
            for label in labels
        ])
        predicted_logits = self.model(input)
        predicted_labels = predicted_logits.argmax(-1)

        loss = F.cross_entropy(predicted_logits, expected_logits)
        error = 100. * (labels != predicted_labels).float().mean()

        return {
            "loss": loss,
            "error": error,
            "accuracy": 100. - error,
        }

    def get_logits(self, label: int):
        if label not in self._class_logits_map:
            logits = [self._off_value] * self._num_classes
            logits[label] = self._on_value

            self._class_logits_map[label] = torch.Tensor(logits).to(self.device)
        return self._class_logits_map[label]

    def write_step(self):
        self.model.eval()
        def _get_prediction_image(iterable, num_images_per_label: int = 16, num_max_batches: int = 10):
            image_map = {}
            num_full_per_label = 0
            image_shape = None
            for batch_idx, batch in enumerate(iterable):
                images, labels = batch
                image_shape = images[0].shape

                predicted_logits = self.model(images)
                predicted_labels = predicted_logits.argmax(-1)

                for image, label, predicted_label in zip(images, labels, predicted_labels):
                    predicted_label = int(predicted_label)
                    if predicted_label not in image_map:
                        image_map[predicted_label] = []
                    if len(image_map[predicted_label]) < num_images_per_label:
                        image_map[predicted_label].append((image.detach().cpu().float(), label))
                        if len(image_map[predicted_label]) >= num_images_per_label:
                            num_full_per_label += 1

                if num_full_per_label >= self._num_classes or batch_idx >= num_max_batches:
                    break

            grid = []
            labels = []
            for idx in range(num_images_per_label):
                for label in range(self._num_classes):
                    if label in image_map and len(image_map[label]) > idx:
                        image, true_label = image_map[label][idx]
                        if true_label == label:
                            labels.append("")
                        else:
                            labels.append(f"{true_label} : {label}")
                    else:
                        image = torch.zeros(image_shape)
                        labels.append("")

                    grid.append(image)

            return make_grid_labeled(tensor=grid, labels=labels, nrow=self._num_classes, padding=2)

        self.log_image("image_matrix_train", _get_prediction_image(self.iter_training_batches()))
        self.log_image("image_matrix_validation", _get_prediction_image(self.iter_validation_batches()))
