from typing import List, Dict, Any, Generator


def iter_matrix_permutations(
        matrix: Dict[str, List[Any]],
        exclude_keys: List[str] = tuple()
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
                for data in iter_matrix_permutations(matrix, exclude_keys + [key]):
                    yield {
                        **entry,
                        **data,
                    }
                    yielded = True

                if not yielded:
                    yield entry

            break
