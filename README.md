# Jupyter Inheritance

_Inherit Jupyter Kernels_

You have a notebook `base.ipynb` with a cell 

```python
import os
from datetime import datetime

class Test:
    msg = "Hey!"

def add(x, y):
    return x + y
    
test = Test()
now = datetime.now()
```

that has been executed. You can create a new notebook and run the following code:

```python
from jupyter_inheritance import inherit_from
inherit_from("base.ipynb")

assert add(1, 4) == 5
assert isinstance(test, Test)

print(test.msg)
print(now)  # same value as `now` in `base.ipynb`!
print(os.listdir("."))
```

The `base.ipynb` content is not executed from scratch in the new notebook,
all the existing objects are copied directly from `base.ipynb` kernel. This
ensures that everything stays exactly the same (e.g. timestamps, random numbers,
responses from externals APIs).

You can even do mixins!

```python
from jupyter_inheritance import inherit_from
for notebook in ("base_1.ipynb", "base_2.ipynb"):
    inherit_from(notebook)
```

## Installation

Just the usual

```bash
pip install jupyter-inheritance
```
