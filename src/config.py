"""
Either define those variables in the environment or
create a `.env` file in the project root:

```
BIG_DATASETS_PATH=/media/user/my-big-device/datasets
```
"""
from pathlib import Path

import decouple


BIG_DATASETS_PATH = Path(decouple.config(
    "BIG_DATASETS_PATH",
    Path("~/datasets"),
)).expanduser()
