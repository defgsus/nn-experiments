from typing import Any, List, Optional, Tuple, Dict, Generator, Iterable


def param_make_tuple(
        value: Any,
        length: int,
        name: Optional[str] = None,
        arg_is_tuple: bool = False,
) -> Tuple[Any]:
    return tuple(param_make_list(value, length, name, arg_is_tuple=arg_is_tuple))


def param_make_list(
        value: Any,
        length: int,
        name: Optional[str] = None,
        arg_is_tuple: bool = False,
) -> List[Any]:
    """
    Convert any value that is not a list (or tuple) to a list of defined length
    by repeating the value `length` times.

    If `value` is a list (or tuple) then it's length is checked and a ValueError
    is raised when unmatched.
    """
    if not isinstance(value, (list, tuple)):
        value_list = [value] * length
    else:
        if arg_is_tuple and isinstance(value, tuple):
            value_list = [value] * length
        else:
            value_list = list(value)

        if len(value_list) != length:
            if not name:
                name = ""
            else:
                name = f"{name} of "
            raise ValueError(f"Expected {name}length {length}, got {len(value_list)}")

    return value_list


def iter_parameter_permutations(
        matrix: Dict[str, Iterable[Any]],
        exclude_keys: Iterable[str] = tuple()
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterates through all items of a `key -> [values]` type dictionary.

    e.g.

        matrix = {
            "a": [0, 1],
            "b": [10, 11],
            "c": [100],
        }
        list(iter_matrix_permutations(matrix))

        [{'a': 0, 'b': 10, 'c': 100},
         {'a': 0, 'b': 11, 'c': 100},
         {'a': 1, 'b': 10, 'c': 100},
         {'a': 1, 'b': 11, 'c': 100}]

    :return: Generator of dict
    """
    exclude_keys = list(exclude_keys)

    for key in matrix.keys():
        if key not in exclude_keys:
            values = matrix[key]

            for v in values:
                entry = {key: v}
                yielded = False
                for data in iter_parameter_permutations(matrix, exclude_keys + [key]):
                    yield {
                        **entry,
                        **data,
                    }
                    yielded = True

                if not yielded:
                    yield entry

            break
