# Signed Distance Field objects via numpy



Scratch for a config language:

```yaml
sources:
  - name: source1
    edges:
      radius: .1
modules:
  - name: mask1
    source: source1
    abs: true
    transform:
      - translate: [0, 1]
      - noise_warp:
          amount: .1
          freq: 4

composition: |
  np.pow(mask1, 2.)
```

Would be much too complicated, i guess. Best use either python or a fully graphical environment.
The latter would, again, be complicated.