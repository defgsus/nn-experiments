import math
import random
from typing import Optional, Callable, List, Tuple, Iterable, Generator, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

from ..base_iterable import BaseIterableDataset


class TextQABaseIterableDataset(BaseIterableDataset):
    """
    Yields '<question> : <answer>', '<question> : ??????' (masked with 0)

    """
    def __init__(
            self,
            count: int,
            separator: str = ": ",
            fixed_question_width: Optional[int] = None,
            fixed_answer_width: Optional[int] = None,
            seed: Optional[int] = None,
            exclude: Optional[Iterable[str]] = None,
            with_masked: bool = False,
    ):
        super().__init__()
        self._count = count
        self._separator = separator
        self._fixed_question_width = fixed_question_width
        self._fixed_answer_width = fixed_answer_width
        self._seed = seed
        self._exclude = None if exclude is None else set(exclude)
        self._with_masked = with_masked

    def iter_question_answer(self, rng: random.Random) -> Generator[Tuple[str, str], None, None]:
        raise NotImplementedError

    @classmethod
    def create_train_and_validation_set(
            cls,
            train_count: int,
            validation_count: int,
            validation_seed: int,
            **kwargs,
    ):
        """
        Create a train and validation dataset

        :param train_count: int
        :param validation_count: int
        :param validation_seed: int
        :param kwargs:
            Any additional parameters.
            If a parameter starts with `validation_`, the `validation_` part is removed
            and the parameter is applied to the validation set, only!
        :return:
        """
        validation_kwargs = kwargs.copy()
        for key in list(validation_kwargs.keys()):
            if key.startswith("validation_"):
                validation_kwargs[key[11:]] = validation_kwargs.pop(key)
                kwargs.pop(key)

        val_set = cls(
            count=validation_count,
            seed=validation_seed,
            **validation_kwargs,
        )
        val_questions = [
            i if isinstance(i, str) else i[0]
            for i in val_set
        ]
        train_set = cls(
            count=train_count,
            exclude=val_questions,
            **kwargs,
        )
        return train_set, val_set

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Generator[Union[str, Tuple[str, str]], None, None]:
        if self._seed is None:
            rng = random
        else:
            rng = random.Random(self._seed)

        num = 0
        for question, answer in self.iter_question_answer(rng):

            if self._fixed_question_width:
                question = question.ljust(self._fixed_question_width)[:self._fixed_question_width]
            if self._fixed_answer_width:
                answer = answer.ljust(self._fixed_answer_width)[:self._fixed_answer_width]

            text = f"{question}{self._separator}{answer}"

            if self._exclude and text in self._exclude:
                continue

            if not self._with_masked:
                yield text
            else:
                masked_text = f"{question}{self._separator}" + "\0" * len(answer)
                yield text, masked_text

            num += 1
            if num >= self._count:
                break


class TextQAMathIterableDataset(TextQABaseIterableDataset):
    """
    Yields things like '3 + 4 = 7'
    """
    def __init__(
            self,
            count: Optional[int] = None,
            num_operations: Union[int, Tuple[int, int]] = 1,
            max_number: int = 10,
            operators: Iterable[str] = ("+",),
            fixed_answer_width: Optional[int] = None,
            seed: Optional[int] = None,
            exclude: Optional[Iterable[str]] = None,
            with_masked: bool = False,
            **kwargs,
    ):
        self._operators = list(operators)
        if count is None:
            count = (max_number ** (num_operations + 1)) * (num_operations ** len(self._operators))
        super().__init__(
            count=count, seed=seed, exclude=exclude, with_masked=with_masked, **kwargs,
            fixed_answer_width=fixed_answer_width,
            separator="=",
        )
        self._num_operations = num_operations
        self._max_number = max_number

    def iter_question_answer(self, rng: random.Random) -> Generator[Tuple[str, str], None, None]:
        while True:
            seq = [str(rng.randint(0, self._max_number))]

            nops = self._num_operations
            if isinstance(nops, (list, tuple)):
                nops = rng.randint(*nops)

            for j in range(nops):
                seq.append(
                    rng.choice(self._operators)
                )
                seq.append(
                    str(rng.randint(0, self._max_number))
                )

            expression = "".join(seq)
            result = str(eval(expression))
            yield expression, result


class TextSelectiveCopyingIterableDataset(TextQABaseIterableDataset):
    """
    Like described in the mamba paper https://arxiv.org/abs/2312.00752

    Yields things like 'A  BC D  : ABCD'
    """
    def __init__(
            self,
            count: int,
            num_items: int = 4,
            area: int = 10,
            seed: Optional[int] = None,
            exclude: Optional[Iterable[str]] = None,
            with_masked: bool = False,
    ):
        assert area >= num_items
        super().__init__(
            count=count, seed=seed, exclude=exclude, with_masked=with_masked,
            fixed_question_width=area,
            fixed_answer_width=num_items,
        )
        self._count = count
        self._num_items = num_items
        self._area = area
        self._seed = seed
        self._exclude = None if exclude is None else set(exclude)
        self._with_masked = with_masked

    def iter_question_answer(self, rng: random.Random) -> Generator[Tuple[str, str], None, None]:
        while True:
            area = [" "] * self._area
            for item_idx in range(self._num_items):
                while True:
                    x = rng.randrange(len(area))
                    if area[x] == " ":
                        area[x] = chr(ord('A') + item_idx)
                        break

            question = "".join(area)
            answer = question.replace(" ", "")

            yield question, answer


class TextQAProgramIterableDataset(TextQABaseIterableDataset):
    """
    Yields things like

        ABCD, 0>1 = BACD
    """
    # operators and probabilities
    DEFAULT_OPERATORS = {
        ">": 1.,
        "-": 1/3,
        "+": 1/3,
    }

    def __init__(
            self,
            count: int,
            input_length: Union[int, Tuple[int, int]] = 4,
            num_items: Union[int, Tuple[int, int]] = 26,
            num_operations: Union[int, Tuple[int, int]] = 3,
            operators: Optional[Dict[str, float]] = None,
            fifo_stack: bool = False,
            seperator: str = " ",
            seed: Optional[int] = None,
            exclude: Optional[Iterable[str]] = None,
            with_masked: bool = False,
            with_params: bool = False,
    ):
        super().__init__(
            count=count, seed=seed, exclude=exclude, with_masked=with_masked,
            separator=":",
            fixed_answer_width=max(input_length) if isinstance(input_length, (tuple, list)) else input_length,
        )
        self._count = count
        self._input_length = input_length
        self._num_items = num_items
        self._num_operations = num_operations
        self._operators = operators or self.DEFAULT_OPERATORS
        self._fifo_stack = fifo_stack
        self._with_params = with_params
        self._seperator = seperator

    def iter_question_answer(self, rng: random.Random) -> Generator[Tuple[str, str], None, None]:
        duplicates_set = set()
        while True:

            input_length = self._input_length
            if isinstance(input_length, (tuple, list)):
                input_length = rng.randint(*input_length)

            num_items = self._num_items
            if isinstance(num_items, (tuple, list)):
                num_items = rng.randint(*num_items)

            num_ops = self._num_operations
            if isinstance(num_ops, (tuple, list)):
                num_ops = rng.randint(*num_ops)

            items = [chr(ord('A') + i) for i in range(num_items)]
            rng.shuffle(items)
            cells = items[:input_length]
            program_input = cells.copy()

            stack = []
            ops = []
            while cells and len(ops) < num_ops:
                op = rng.choices(
                    list(self._operators.keys()),
                    weights=list(self._operators.values()),
                )[0]
                if op == "-":
                    idx = rng.randrange(len(cells))
                    stack.append(cells.pop(idx))
                    ops.append(f"{op}{idx+1}")
                elif op == "+" and len(stack):
                    idx = rng.randrange(len(cells))
                    cells.insert(idx, stack.pop(0 if self._fifo_stack else -1))
                    ops.append(f"{op}{idx+1}")
                elif op == ">" and len(cells) >= 2:
                    indices = list(range(len(cells)))
                    rng.shuffle(indices)
                    idx1, idx2 = indices[:2]
                    cells[idx1], cells[idx2] = cells[idx2], cells[idx1]
                    ops.append(f"{idx1+1}{op}{idx2+1}")

            question = (
                "".join(program_input) + ":"
                + self._seperator.join(ops)
            )
            if question in duplicates_set:
                continue
            duplicates_set.add(question)

            answer = "".join(cells)

            if not self._with_params:
                yield question, answer
            else:
                yield question, answer, {
                    "num_items": len(program_input),
                    "num_operations": len(ops),
                }
