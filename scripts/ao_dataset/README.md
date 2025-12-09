# Ambient Occlusion Dataset

Renders training images: 

- white fonts on black background
  - in `$PROJECT/datasets/ao/train/source/00xx.png`
  - in `$PROJECT/datasets/ao/validation/source/00xx.png`
- same images with ambient occlusion and shadow
  - in `$PROJECT/datasets/ao/train/target/00xx.png`
  - in `$PROJECT/datasets/ao/validation/target/00xx.png`
- intermediate step (signed distance fields)
  - in `$PROJECT/datasets/ao/train/distance/00xx.pt`
  - in `$PROJECT/datasets/ao/validation/distance/00xx.pt`

```shell
# in project root call
python scripts/ao_dataset/
```

