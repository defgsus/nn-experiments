

# "manifold" decoder

## 2023-11-09

After preliminary experiments, running this setup:

    DalleManifoldAutoencoder(
        shape=(1, 32, 32), 
        vocab_size=128, n_hid=64, n_blk_per_group=1, act_fn=nn.GELU, space_to_depth=True, 
        decoder_n_blk=4, decoder_n_layer=2, decoder_n_hid=64,
    )
    encoder params: 1,725,264
    decoder params: 42,497
    batch_size: 64
    steps: 1M
    learnrate: .0003 AdamW, CosineAnnealingLR 
    
on only 300 (randomly h&v-flipped) images of the RPG-tiles dataset (/scripts/datasets.py).

Besides l2 reconstruction loss there is an extra constraint on the distribution of the encoding:

    loss_batch_std = (.5 - feature_batch.std(0).mean()).abs()
    loss_batch_mean = (0. - feature_batch.mean()).abs()

The three runs add these losses with factor 0.1 (green), 0.001 (orange) and 0.0 (gray). 

![loss plots](./img/ae-manifold-std-constraint.png)

Below are reproduced (right) samples of the orange model. 

![repros](./img/ae-manifold-std-constraint-001-repros.png)

and rendered to 64x64 resolution:
![repros](./img/ae-manifold-std-constraint-001-repros-64.png)

### upgrade decoder

fixed the std/mean loss factor to 0.0001 and increased number of decoder blocks:

    decoder_n_blk=8,  decoder_n_layer=2, decoder_n_hid=128, params: 283,649

plots in x = steps (top) and relative time (bottom):
![loss plots](./img/ae-manifold-std-constraint-plus-b8.png)

The reproductions from the training set look good enough. 
other tiles can hardly be reproduced:

![repros](./img/ae-manifold-std-constraint-0001-b8-l2-repros.png)
![repros](./img/ae-manifold-std-constraint-0001-b8-l2-repros-64.png)


Some (very short) tests with different block/layer settings: 

    (cyan)    decoder_n_blk=8,  decoder_n_layer=2, decoder_n_hid=128, params: 283,649 
    (yellow)  decoder_n_blk=8,  decoder_n_layer=4, decoder_n_hid=128, params: 547,841
    (brown)   decoder_n_blk=16, decoder_n_layer=1, decoder_n_hid=128, params: 285,697 
    (magenta) decoder_n_blk=16, decoder_n_layer=2, decoder_n_hid=128, params: 549,889

![repros](./img/ae-manifold-std-constraint-block-level-compare.png)
