# Project Notes / Daily Learnings

## Python

### General

 - You can use tuples as keys for dicts
 - This means that you can do

 ```python

 d = dict()

 def foo(*args):
    d[args] = "bar"
```

### Memoization

https://dbader.org/blog/python-memoization

Built in memoization can be done with

```python
from functools import lru_cache

@lru_cache(maxsize=128, typed=False)
def foo(bar, bif):
    # Expensive stuff here
    return answer
```

maxsize defines the number of items stored at any one time
typed creates distinct keys for 3 and 3.0

get cache info with foo.cache_info()
clear cache with foo.cache_clear()

Note: This is implemented using a linked list! Neat.

### Scheduling Work

check out the python `schedule` module by Dan Bader to run things periodically
still need to launch a python program *shrug*

https://github.com/dbader/schedule



