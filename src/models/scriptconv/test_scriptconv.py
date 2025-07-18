from typing import Tuple
from textwrap import dedent

from src.tests.base import *

from src.models.scriptconv import ScriptConvModel
from src.util.module import num_module_parameters


class TestScriptConv(TestBase):

    def assert_model(
            self,
            script: str,
            input_shape: Tuple[int, int, int],
            expected_output_shape: Tuple[int, ...],
            expected_repr: str,
    ):
        model = ScriptConvModel(
            script=script,
            input_shape=input_shape,
        )

        expected_repr = dedent(expected_repr).strip()
        model_repr = repr(model).strip()

        self.assertEqual(
            expected_repr,
            model_repr,
            f"\nExpected:\n{expected_repr}\n\nGot:\n{model_repr}"
        )

        inp = torch.ones(8, *input_shape)
        outp = model(inp)
        self.assertEqual(
            expected_output_shape,
            outp.shape[1:],
        )

        print(f"\n{script}\nparams: {num_module_parameters(model):,}")

    def test_conv_attr(self):
        self.assert_model(
            "16xk2s3d4p5arelu-32x5",
            (3, 100, 100),
            (32, 32, 32),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 16, kernel_size=(2, 2), stride=(3, 3), padding=(5, 5), dilation=(4, 4))
                (1): ReLU()
                (2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
              )
            )
            """,
        )

    def test_default_conv_attr(self):
        self.assert_model(
            "k4p5s6d7-16x-20x3d1-24x",
            (3, 450, 450),
            (24, 1, 1),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 16, kernel_size=(4, 4), stride=(6, 6), padding=(5, 5), dilation=(7, 7))
                (1): Conv2d(16, 20, kernel_size=(3, 3), stride=(6, 6), padding=(5, 5))
                (2): Conv2d(20, 24, kernel_size=(4, 4), stride=(6, 6), padding=(5, 5), dilation=(7, 7))
              )
            )
            """,
        )

    def test_float_kernel_size(self):
        self.assert_model(
            "16x-4x(2.x)",
            (3, 100, 100),
            (256, 90, 90),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
                (1): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
                (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
                (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
                (4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
              )
            )
            """,
        )

    def test_residual_loop(self):
        self.assert_model(
            "16x-r(3x(16xk5p2))-32x",
            (3, 100, 100),
            (32, 96, 96),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
                (1): ResidualAdd(
                  (module): Sequential(
                    (0): Conv2d(16, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
                    (1): Conv2d(16, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
                    (2): Conv2d(16, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
                  )
                )
                (2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
              )
            )
            """,
        )

    def test_residual_default_conv(self):
        self.assert_model(
            "k7p3-16x-r(k5p2-2x(16x))-32x",
            (3, 100, 100),
            (32, 100, 100),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
                (1): ResidualAdd(
                  (module): Sequential(
                    (0): Conv2d(16, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
                    (1): Conv2d(16, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
                  )
                )
                (2): Conv2d(16, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
              )
            )
            """,
        )

    def test_batch_norm(self):
        self.assert_model(
            "32x-bn-48x-bn",
            (3, 100, 100),
            (48, 96, 96),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
                (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1))
                (3): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            """,
        )

    def test_pool(self):
        self.assert_model(
            "32x-avgp-16x-maxp4s3d2p1",
            (3, 100, 100),
            (16, 31, 31),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
                (1): AvgPool2d(kernel_size=2, stride=1, padding=0)
                (2): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1))
                (3): MaxPool2d(kernel_size=4, stride=3, padding=1, dilation=2, ceil_mode=False)
              )
            )
            """,
        )

    def test_global_max_pool(self):
        self.assert_model(
            "32x-gmaxp",
            (3, 100, 100),
            (32, ),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
                (1): MaxPool2d(kernel_size=(98, 98), stride=(98, 98), padding=0, dilation=1, ceil_mode=False)
                (2): Flatten(start_dim=-3, end_dim=-1)
              )
            )
            """,
        )

    def test_global_avg_pool(self):
        self.assert_model(
            "32xs2-gavgp",
            (3, 100, 100),
            (32, ),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2))
                (1): AvgPool2d(kernel_size=(49, 49), stride=(49, 49), padding=0)
                (2): Flatten(start_dim=-3, end_dim=-1)
              )
            )
            """,
        )

    def test_dropout(self):
        self.assert_model(
            "32x-do-16x-do.2",
            (3, 100, 100),
            (16, 96, 96),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
                (1): Dropout2d(p=0.5, inplace=False)
                (2): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1))
                (3): Dropout2d(p=0.2, inplace=False)
              )
            )
            """,
        )

    def test_linear(self):
        self.assert_model(
            "32x-fc10",
            (3, 100, 100),
            (10,),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
                (1): Flatten(start_dim=-3, end_dim=-1)
                (2): Linear(in_features=307328, out_features=10, bias=True)
              )
            )
            """,
        )

    def test_linear_2(self):
        self.assert_model(
            "32x-fc1000-bn-do-fc10",
            (3, 100, 100),
            (10,),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
                (1): Flatten(start_dim=-3, end_dim=-1)
                (2): Linear(in_features=307328, out_features=1000, bias=True)
                (3): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): Dropout(p=0.5, inplace=False)
                (5): Linear(in_features=1000, out_features=10, bias=True)
              )
            )
            """,
        )

    def test_linear_residual(self):
        self.assert_model(
            "32x7-2x(fc100)-r(2x(fc100))",
            (3, 100, 100),
            (100,),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1))
                (1): Flatten(start_dim=-3, end_dim=-1)
                (2): Linear(in_features=282752, out_features=100, bias=True)
                (3): Linear(in_features=100, out_features=100, bias=True)
                (4): ResidualAdd(
                  (module): Sequential(
                    (0): Linear(in_features=100, out_features=100, bias=True)
                    (1): Linear(in_features=100, out_features=100, bias=True)
                  )
                )
              )
            )
            """,
        )

    def test_SimpNet(self):
        """
        (Almost) the SimpNet architecture
        https://arxiv.org/abs/1802.06205

        Although i get 5M parameters which is mentioned nowhere in the paper ???
        """
        self.assert_model(
            "arelu-66xk3-bn-3x(128x3-bn)-192x3-bn-maxp2-do.5-4x(192x3-bn)-288x3-bn-maxp2-do.5-288x3-bn-355x3-bn-432x3-bn-gmaxp-fc10",
            (3, 100, 100),
            (10,),
            """
            ScriptConvModel(
              (layers): Sequential(
                (0): Conv2d(3, 66, kernel_size=(3, 3), stride=(1, 1))
                (1): ReLU()
                (2): BatchNorm2d(66, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (3): Conv2d(66, 128, kernel_size=(3, 3), stride=(1, 1))
                (4): ReLU()
                (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (6): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
                (7): ReLU()
                (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (9): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
                (10): ReLU()
                (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (12): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1))
                (13): ReLU()
                (14): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (15): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
                (16): Dropout2d(p=0.5, inplace=False)
                (17): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1))
                (18): ReLU()
                (19): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (20): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1))
                (21): ReLU()
                (22): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (23): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1))
                (24): ReLU()
                (25): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (26): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1))
                (27): ReLU()
                (28): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (29): Conv2d(192, 288, kernel_size=(3, 3), stride=(1, 1))
                (30): ReLU()
                (31): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (32): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
                (33): Dropout2d(p=0.5, inplace=False)
                (34): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1))
                (35): ReLU()
                (36): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (37): Conv2d(288, 355, kernel_size=(3, 3), stride=(1, 1))
                (38): ReLU()
                (39): BatchNorm2d(355, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (40): Conv2d(355, 432, kernel_size=(3, 3), stride=(1, 1))
                (41): ReLU()
                (42): BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (43): MaxPool2d(kernel_size=(72, 72), stride=(72, 72), padding=0, dilation=1, ceil_mode=False)
                (44): Flatten(start_dim=-3, end_dim=-1)
                (45): Linear(in_features=432, out_features=10, bias=True)
              )
            )
            """,
        )
