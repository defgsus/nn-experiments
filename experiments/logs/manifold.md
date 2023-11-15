This is about learning "implicit neural representation",
mainly calculate `position + code -> color`.

# resnet21 for embedding -> ImageManifoldModel

Autoencoder with pre-trained resnet21 (without last avgpool).

```yaml
matrix:
  # compression ratio
  cr: [10]
  opt: ["Adam"]
  lr: [0.001]

  hid: [256]
  # blocks
  bl: [2]
  # layers per block
  lpb: [2]
  act: ["gelu"]

experiment_name: tests/rpg_res5_${matrix_slug}

trainer: experiments.ae.trainer.TrainAutoencoderSpecial

globals:
  SHAPE: (3, 32, 32)
  CODE_SIZE: 32 * 32 // ${cr}

train_set: |
  from experiments.datasets import rpg_tile_dataset 
  rpg_tile_dataset(SHAPE, validation=False, shuffle=True, random_shift=4, random_flip=True)

freeze_validation_set: True
validation_set: |
  from experiments.datasets import rpg_tile_dataset 
  rpg_tile_dataset(SHAPE, validation=True, shuffle=True, limit=500)

batch_size: 64
learnrate: ${lr}
optimizer: ${opt}
scheduler: CosineAnnealingLR
loss_function: l1
max_inputs: 1_000_000

model: |
  from src.models.encoder import resnet
  encoder = resnet.resnet18_open(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
  with torch.no_grad():
      out_shape = encoder(torch.empty(2, *SHAPE)).shape[-3:]
  encoder = nn.Sequential(
      encoder,
      nn.Flatten(1),
      nn.Linear(math.prod(out_shape), CODE_SIZE)
  )
  #for p in encoder.parameters():
  #  p.requires_grad = False
  
  EncoderDecoder(
      encoder,
      ImageManifoldDecoder(
          num_input_channels=CODE_SIZE,
          num_output_channels=SHAPE[0],
          default_shape=SHAPE[-2:],
          num_hidden=${hid},
          num_blocks=${bl},
          num_layers_per_block=${lpb},
          activation="${act}",
      )
  )
```

Increasing the *number of hidden cells*, *blocks* and *layers per block*
did **not** provide a benefit on the 7k rpg tile dataset.
All larger versions performed worse:

![loss plots](./img/ae-manifold-rpg-size.png)

And, actually, it turns out that a simple 3-layer CNN (ks=3, channels=[16, 24, 32])
as the encoder performs much better than the resnet:

```
(encoder): EncoderConv2d(
    (convolution): Conv2dBlock(
      (_act_fn): ReLU()
      (layers): Sequential(
        (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
        (1): ReLU()
        (2): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1))
        (3): ReLU()
        (4): Conv2d(24, 32, kernel_size=(3, 3), stride=(1, 1))
        (5): ReLU()
      )
    )
    (linear): Linear(in_features=21632, out_features=102, bias=True)
  )
```

![loss plots](./img/ae-manifold-rpg-simple-cnn.png)

#### Update

Everything i tried in the last couple of days is performing 
worse, e.g. changing the pos-embedding frequencies, 
using FFTs in some way and incresing the encoder params.

By the way, running the tests for 1M steps (about 20 epochs
with the current RPG tile dataset) might not be 
enough either... but i'm actually looking for methods that
enhance performance already before 1M steps. For the sake
of logging: 

![loss plots](./img/ae-manifold-rpg-7M.png)

The gray line is the reference with simple CNN encoder from 
above and yellow has increased the encoder channels from 
`(16, 24, 32)` to `(24, 32, 48)` which, if course!, performed
a little worse in the beginning. Made it run for 7M steps, 
which is over 200 epochs (on a randomly flipped and shifted 
dataset). It went below 0.04 l1 validation loss, but this
is still bad:

![repros](./img/ae-manifold-rpg-7M-repros.png)

The idea behind using the implicit generation is to be
able to increase the resolution, but if it already looks
blurry in the original resolution...
