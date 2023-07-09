import random
import torch
import torch.nn.functional as F
import torch.utils.data

from src.models.recursive import RecursiveUnit
from src import console


# class DataSource(torch.utils.data.Dataset)


def train_unit(
        model: RecursiveUnit,
        batch_size: int = 50,
        num_epochs: int = 100,
        n_rec_iter: int = 30,
):
    console.print_tensor_2d(model.recursive_weights)

    optimizers = [
        torch.optim.SGD(model.parameters(), lr=.005, momentum=0., weight_decay=0.)
    ]

    dataset = torch.utils.data.TensorDataset(
        torch.rand(1000, model.n_cells) * 2. - 1.
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    def dump_stats():
        with torch.no_grad():
            w_min, w_max = float(model.recursive_weights.min()), float(model.recursive_weights.max())
            complexity = float(model.calc_complexity(dataset[0][0], n_iter=n_rec_iter))

        prompt = f"epoch {epoch} step {step}:"
        print(f"{prompt}: loss {loss:8} weights {w_min} - {w_max}")
        #print(f"#{step:5} weights:")
        #console.print_tensor_2d(model.recursive_weights)
        print(f"{prompt} state train:")
        dump_state_train(model, dataset[0][0], n_iter=n_rec_iter)
        print(f"{prompt} complexity:", complexity)

    last_loss = None
    last_complexity = None
    step = 0
    for epoch in range(num_epochs):
        for x, in dataloader:

            complexity = model.calc_complexity(x, n_iter=n_rec_iter, start_iter=4)
            loss = .2 - complexity
            loss += .3 * model.hebbian_loss(x, n_iter=random.randint(1, n_rec_iter))

            model.zero_grad()
            loss.backward()
            for opt in optimizers:
                opt.step()

            step += x.shape[0]
            loss = float(loss)
            complexity = float(complexity)
            if (
                    #or (last_loss is None or abs(loss - last_loss) > .2)
                    (last_complexity is None or (complexity - last_complexity) > .1)
            ):
                last_loss = loss
                last_complexity = complexity

                dump_stats()

            #if step % 20 == 0:
            #    console.print_tensor_2d(model.recursive_weights)

    dump_stats()


def dump_state_train(unit: RecursiveUnit, x: torch.Tensor, n_iter: int = 10, threshold: float = .5):
    assert x.ndim == 1, x.ndim
    with torch.no_grad():
        states = [x.unsqueeze(0)]
        for i in range(n_iter + 1):
            x = unit.single_pass(x)
            states.append(x.unsqueeze(0))

        states = torch.cat(states)

        console.print_tensor_2d(states)


def debug_complexity(
        model: RecursiveUnit,
        n_iter: int = 30,
):
    with torch.no_grad():
        x = torch.rand(1, model.n_cells) * 2. - 1.

        for i in range(1000):
            model.init_weights()
            c = float(model.calc_complexity(x, n_iter=n_iter, start_iter=5))
            if c > .1:
                dump_state_train(model, x[0], n_iter=n_iter)
                print("complexity:", c)



def main():
    unit = RecursiveUnit(
        n_cells=20,
        #act_fn=torch.tanh,
        act_fn=torch.sin,
    )
    unit.init_weights()

    #debug_complexity(unit)
    train_unit(unit)


if __name__ == "__main__":
    main()
