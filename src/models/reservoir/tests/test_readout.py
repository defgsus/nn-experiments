import unittest

import torch

from src.models.reservoir import *


class TestReservoirReadout(unittest.TestCase):

    def get_predict_sequence(self, timesteps: int, forward_steps: int = 1, channels: int = 1):
        t = torch.linspace(0, 10 * torch.pi, timesteps + forward_steps)
        seq = torch.sin(t)[None, :, None]
        if channels > 1:
            seq = seq.expand(1, -1, channels)
        return (
            seq[:, :-forward_steps, :],
            seq[:, forward_steps:, :],
        )


    def run_readout(self, esn: ReservoirReadout, num_inputs: int, num_states: int):
        self.assertEqual(esn.reservoir.num_inputs, num_inputs)
        self.assertEqual(esn.reservoir.num_states, num_states)

        # run for T steps
        state = esn.run_reservoir(steps=11)
        self.assertEqual(torch.Size((1, 11, num_states)), state.shape)

        # train to predict next sequence value
        input, target = self.get_predict_sequence(100, channels=num_inputs)
        error_l1, error_l2 = esn.fit(input, target)

        self.assertLessEqual(error_l1, 0.05)
        self.assertLessEqual(error_l2, 0.5)

    def test_100_normal_reservoir(self):
        esn = ReservoirReadout(
            reservoir=Reservoir(
                num_inputs=1,
                num_cells=100,
            )
        )
        self.run_readout(esn, 1, 100)

    def test_200_parallel_reservoirs(self):
        esn = ReservoirReadout(
            reservoir=ParallelReservoirs(
                Reservoir(
                    num_inputs=2,
                    num_cells=100,
                ),
                Reservoir(
                    num_inputs=1,
                    num_cells=110,
                ),
            )
        )
        self.run_readout(esn, 3, 210)

    def test_210_parallel_reservoirs_shared_input(self):
        esn = ReservoirReadout(
            reservoir=ParallelReservoirs(
                Reservoir(
                    num_inputs=1,
                    num_cells=100,
                ),
                Reservoir(
                    num_inputs=1,
                    num_cells=110,
                ),
                share_inputs=True,
            )
        )
        self.run_readout(esn, 1, 210)

    def test_300_sequential_reservoirs(self):
        esn = ReservoirReadout(
            reservoir=SequentialReservoirs(
                Reservoir(
                    num_inputs=2,
                    num_cells=50,
                ),
                Reservoir(
                    num_inputs=50,
                    num_cells=100,
                ),
            )
        )
        self.run_readout(esn, 2, 150)
