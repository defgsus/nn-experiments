import argparse
import gzip
import json
import random
import math
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.bd_rl.bdenv import BoulderDashEnvironment
from experiments.bd_rl.bdmodels import BoulderDashPredictModel, BoulderDashPolicyModel
from experiments.bd_rl.bdtrainer import BoulderDashRLTrainer


DIRECTORY = Path(__file__).resolve().parent / "data"


def parse_args():
    main_parser = argparse.ArgumentParser()

    main_parser.add_argument(
        "-s", "--map-size", type=int, default=16,
        help="width and height of map",
    )

    subparsers = main_parser.add_subparsers()

    # -----------

    parser = subparsers.add_parser("train", help="Run RL training")
    parser.set_defaults(command="train")

    # -----------

    parser = subparsers.add_parser("create_examples", help="Run RL training")
    parser.set_defaults(command="create_examples")

    parser.add_argument(
        "filename", type=str,
        help="filename to store examples, `.ndjson.gz` will be added if needed",
    )

    parser.add_argument(
        "-c", "--count", type=int, default=10,
        help="number of examples to render",
    )

    # -----------

    parser = subparsers.add_parser("test_action_eval_model", help="Run RL training")
    parser.set_defaults(command="test_action_eval_model")

    parser.add_argument(
        "model_file", type=str,
        help="filename of model.pt",
    )

    return vars(main_parser.parse_args())


def main(command: str, **kwargs):
    command_func = globals().get(f"command_{command}")
    if not callable(command_func):
        print(f"Invalid command '{command}'")
        exit(1)

    command_func(**kwargs)


def command_train(
        map_size: int,
):
    environment = BoulderDashEnvironment(
        shape=(map_size, map_size)
    )

    policy_model = create_policy_model(map_size)
    print(f"\npolicy model:\n{policy_model}")
    target_model = create_policy_model(map_size)
    target_model.load_state_dict(policy_model.state_dict())
    target_model.eval()

    trainer = BoulderDashRLTrainer(
        environment=environment,
        policy_model=policy_model,
        target_model=target_model,
    )
    #trainer.eval()
    trainer.train()


def command_create_examples(
        map_size: int,
        filename: str,
        count: int,
):
    environment = BoulderDashEnvironment(shape=(map_size, map_size))

    with gzip.open(get_example_filename(filename), "wt") as fp:
        i = 0
        try:
            with tqdm(desc="creating examples", total=count) as progress:
                while i < count:
                    environment.reset()

                    skip_this = False
                    for _ in range(random.randrange(3) * random.randrange(3)):
                        if environment.bd.apply_physics() == environment.bd.RESULTS.PlayerDied:
                            skip_this = True
                            break
                    if skip_this:
                        continue

                    state = environment.bd.to_array()

                    action_results = []
                    for action in environment.all_actions:
                        environment.push_state()
                        result, reward, terminated = environment.act(action)
                        next_state = environment.bd.to_array()
                        if result != environment.bd.RESULTS.PlayerDied:
                            closest_diamond = environment.bd.path_to_closest_diamond()
                        else:
                            closest_diamond = None
                        environment.pop_state()

                        action_results.append({
                            "action": action,
                            "next_state": next_state,
                            "result": result,
                            "reward": reward,
                            "terminated": terminated,
                            "closest_diamond": closest_diamond,
                        })

                    fp.write(json.dumps({
                        "state": state,
                        "closest_diamond": environment.bd.path_to_closest_diamond(),
                        "actions": action_results,
                    }))
                    fp.write("\n")
                    i += 1
                    progress.update()

        except KeyboardInterrupt:
            pass


def command_test_action_eval_model(
        map_size: int,
        model_file: str,
):
    from experiments.bd.model import BoulderDashActionPredictModel, BDModelInput

    model = BoulderDashActionPredictModel(
        shape=(map_size, map_size), num_hidden=16, kernel_size=3, num_layers=6, act="gelu"
    )
    data = torch.load(model_file)
    model.load_state_dict(data["state_dict"])

    def _eval_next_action():
        num_actions = environment.bd.ACTIONS.count()

        action_batch = []
        for name, value in environment.bd.ACTIONS.items():
            action = [-1.] * num_actions
            action[value] = 1.
            action_batch.append(action)

        input = BDModelInput(
            state=environment.state().unsqueeze(0).repeat(num_actions, 1, 1, 1),
            action=torch.Tensor(action_batch),
        )
        output = model.forward(input)
        best_action = int(output.reward.reshape(-1).argmax())

        for name, value in environment.bd.ACTIONS.items():
            print(f"{name:10}: predicted reward {'*' if value == best_action else ' '} {output.reward[value].item()}")

        return best_action

    environment = BoulderDashEnvironment(shape=(map_size, map_size))

    while True:
        environment.bd.dump(ansi_colors=True)

        action = _eval_next_action()

        cmd = input("\n> ").lower()

        if cmd == "q":
            break
        elif cmd == "r":
            environment.reset()
            continue
        elif cmd == "w":
            action = environment.bd.ACTIONS.Up
        elif cmd == "a":
            action = environment.bd.ACTIONS.Left
        elif cmd == "s":
            action = environment.bd.ACTIONS.Down
        elif cmd == "d":
            action = environment.bd.ACTIONS.Right

        result, reward, terminated = environment.act(action)
        print("exec:", result, reward, terminated)


def create_policy_model(map_size: int):
    prediction_model = BoulderDashPredictModel(
        shape=(map_size, map_size),
    )
    # data = torch.load("./checkpoints/bd/bd-predict-01-sml_opt-AdamW_lr-0.003_ks-3_hid-16_l-6_act-gelu/best.pt")
    data = torch.load(DIRECTORY / "predict-model-ks3-hid16-l6.pt")
    prediction_model.load_state_dict(data["state_dict"])

    policy_model = BoulderDashPolicyModel(
        shape=(map_size, map_size),
        prediction_model=prediction_model,
    )
    return policy_model


def get_example_filename(filename: str):
    filename_l = filename.lower()
    if not filename_l.endswith(".ndjson.gz"):
        filename = f"{filename}.ndjson.gz"
    return DIRECTORY / filename


if __name__ == "__main__":
    main(**parse_args())
