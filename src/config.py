"""
Either define those variables in the environment or
create a `.env` file in the project root:

```
BIG_DATASETS_PATH=/media/user/my-big-device/datasets
```
"""
from pathlib import Path

import decouple


PROJECT_PATH = Path(__file__).resolve().parent.parent


BIG_DATASETS_PATH = Path(decouple.config(
    "BIG_DATASETS_PATH",
    Path("~/datasets"),
)).expanduser()


SMALL_DATASETS_PATH = Path(decouple.config(
    "SMALL_DATASETS_PATH",
    Path("~/prog/data"),
)).expanduser()
