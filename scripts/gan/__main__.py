import importlib
from pathlib import Path
import argparse

from scripts.gan.trainer import GANTrainer, GANTrainerSettings
from src.util.module import num_module_parameters


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment_file", type=str,
        help="Filename of the experiment, e.g. `exp1`"
    )


    return vars(parser.parse_args())


def main(
        experiment_file: str,
):
    module = importlib.import_module(experiment_file)

    generator = module.get_generator()
    discriminator = module.get_discriminator()

    small_path = f"gan/{experiment_file}"
    base_path = GANTrainer.PROJECT_PATH / "runs"
    run_index = 1
    while (base_path / small_path / str(run_index)).exists():
        run_index += 1

    settings = GANTrainerSettings(
        path=Path(small_path) / str(run_index),
        generator=generator,
        discriminator=discriminator,
        train_dataset=module.get_dataset(validation=False),
        validation_dataset=module.get_dataset(validation=True),
    )
    module_settings = module.get_settings()
    for key, value in vars(module_settings).items():
        if hasattr(settings, key):
            if value is not None:
                setattr(settings, key, value)

    print(settings)
    #print("generator:\n")
    #print(generator)
    #print(f"\ndiscriminator:\n")
    #print(discriminator)
    print(f"\ngenerator params:     {num_module_parameters(generator):,}")
    print(f"discriminator params: {num_module_parameters(discriminator):,}\n")

    trainer = GANTrainer(settings)
    trainer.train()



if __name__ == "__main__":
    main(**parse_args())

