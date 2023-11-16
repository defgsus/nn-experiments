

# variational auto-encoder on RPG Tile dataset

There is a *deep* love/hate relationships with neural networks.
Why the heck do i need to train a small network like this

```python
model = VariationalAutoencoderConv(
    shape=(3, 32, 32), channels=[16, 24, 32], kernel_size=5, 
    latent_dims=128,
)

optimizer = Adam(model.parameters(), lr=.0001, weight_decay=0.000001)
```

for 10 hours and it still does not reach the optimum?

![loss plots](./img/vae-rpg-conv16-24-32-40M.png)

And how could one tell after 30 minutes where this is going
to go? The plot shows the l1 validation loss (right) over
**1700 epochs!** Why does this network need to look at 
things 1700 times???

Well, it's a complicated dataset, for sure.

![reproductions](./img/vae-rpg-conv16-24-32-40M-repros.png)

But i feel there is something wrong in the method. 
This *backpropagation gradient descent*, although
mathematically grounded, feels like a brute-force approach.

