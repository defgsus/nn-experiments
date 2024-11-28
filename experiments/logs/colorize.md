


```yaml
experiment_name: img2img/colorize-resconv-${matrix_slug}

matrix:
  ds:
    - "cifar10"
    #- "all"
  space:
    - "rgb"
    - "hsv"
    - "hsv2"
    - "lab"
    - "xyz"
    - "ycbcr"
  bs: [64]
  opt: ["AdamW"]
  lr: [.0003]
  l:
    - 11
  ks:
    - 3
  pad:
    - 1
  ch:
    - 32
  stride:
    - 1
  act:
    - gelu

trainer: experiments.img2img.trainer.TrainImg2Img
first_arg_is_transforms: |
  [
    VT.Grayscale(),  
    # RandomCropHalfImage(),
  ]
# histogram_loss_weight: 100.

train_set: |
  if "${ds}" == "all":
      ds = all_image_patch_dataset(shape=SHAPE).skip(5000)
  else:
      ds = ${ds}_dataset(train=True, shape=SHAPE, interpolation=False)
  ds

validation_set: |
  if "${ds}" == "all":
      ds = all_image_patch_dataset(shape=SHAPE).limit(5000)
  else:
      ds = ${ds}_dataset(train=False, shape=SHAPE, interpolation=False)
  ds

batch_size: ${bs}
learnrate: ${lr}
optimizer: ${opt}
scheduler: CosineAnnealingWarmupLR
loss_function: l1
max_inputs: 1_000_000
num_inputs_between_validations: 100_000
freeze_validation_set: True


globals:
  SHAPE: (3, 32, 32)

model: |
  from experiments.denoise.resconv import ResConv
  from src.functional import colorconvert
  
  class Module(nn.Module):
      def __init__(self):
          super().__init__()
          self.module = ResConv(
              in_channels=3,
              out_channels=SHAPE[0],
              num_layers=${l},
              channels=${ch},
              stride=${stride},
              kernel_size=${ks},
              padding=${pad},
              activation="${act}",
              activation_last_layer=None,
          )
  
      def forward(self, x):
          y = x.repeat(1, 3, 1, 1)
          if "${space}" == "hsv2":
              y = rgb_to_hsv(y)
          elif "${space}" != "rgb":
              y = colorconvert.rgb2${space}(y)
          y = self.module(y)
          if "${space}" == "hsv2":
              y = hsv_to_rgb(y)
          elif "${space}" != "rgb":
              y = colorconvert.${space}2rgb(y)
          return y
          
  Module()

```


| ds      | space   |   bs | opt   |     lr |   l |   ks |   pad |   ch |   stride | act   |   validation loss (1,000,000 steps) | model params   |   train time (minutes) | throughput   |
|:--------|:--------|-----:|:------|-------:|----:|-----:|------:|-----:|---------:|:------|------------------------------------:|:---------------|-----------------------:|:-------------|
| cifar10 | hsv2    |   64 | AdamW | 0.0003 |  11 |    3 |     1 |   32 |        1 | gelu  |                           0.0454638 | 187,945        |                   7.72 | 2,159/s      |
| cifar10 | lab     |   64 | AdamW | 0.0003 |  11 |    3 |     1 |   32 |        1 | gelu  |                           0.0454897 | 187,945        |                   9.32 | 1,788/s      |
| cifar10 | rgb     |   64 | AdamW | 0.0003 |  11 |    3 |     1 |   32 |        1 | gelu  |                           0.0457786 | 187,365        |                   4.06 | 4,105/s      |
| cifar10 | ycbcr   |   64 | AdamW | 0.0003 |  11 |    3 |     1 |   32 |        1 | gelu  |                           0.0466509 | 187,945        |                   8.1  | 2,056/s      |
| cifar10 | xyz     |   64 | AdamW | 0.0003 |  11 |    3 |     1 |   32 |        1 | gelu  |                           0.0493999 | 187,945        |                   8.67 | 1,921/s      |
| cifar10 | hsv     |   64 | AdamW | 0.0003 |  11 |    3 |     1 |   32 |        1 | gelu  |                           0.147812  | 187,945        |                   8.29 | 2,010/s      |
