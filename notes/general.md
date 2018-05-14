# Project Notes / Daily Learnings

## Python

### Packaging / Setup / Distribution

 - Kenneth Reitz has a number of excellent projects that are worth emulating.
 - The most important file for pip is setup.py
   - KR maintains a easy to use setup.py file that supports uploading!

### Linting

 - flake8 is a solid modern linter that works well with Sublime3
    - install flake8 executable locally `pip install flake8`
    - install Sublimelinter package for sublime
    - install Sublimelinter-flake8 plugin for sublimelinter
    - *restart sublime*

### General

 - You can use tuples as keys for dicts
 - This means that you can do

 ```python

d = dict()


def foo(*args):
    d[args] = "bar"
```

### Internals / Magic

 - id() will return a unique for any python object
    - this is very useful for determining if two things are really the same

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



