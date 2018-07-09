# Project Notes / Daily Learnings

## Testing

Basic unit tests - pytest
Test coverage - coverage + pytest-cov
Fuzz testing? - hypothesis
Multi-config testing - tox (uses virtual envs)

## Tensorflow

#### Pushback on more flexible core RNNs

https://github.com/tensorflow/tensorflow/pull/2767

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

### Stats Related

Create a histogram of all the values seen

```python
from collections import Counter
Counter(<list of values>)
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

### Asyncio

#### Create an event loop

```python
import asyncio

loop = asyncio.get_event_loop()
```

#### Enqueue a function as part of the event loop

```python
loop = asyncio.get_event_loop()
async def foo():
    print('foo')
    asyncio.ensure_future(foo())


asyncio.ensure_future(foo())
loop.run_forever()
```
Note: You can't just recurse in python forever so you need to add new tasks
to the event loop instead of re-calling a function

Note: You can't call *synchronous* functions like this as that will hit the
recursion limit as well.

It *appears* that you can call

```python
loop.create_task()
```

interchangeably with 

```python
asyncio.ensure_future()
```

so the above could be
```python
loop = asyncio.get_event_loop()
async def foo():
    print('foo')
    loop.create_task(foo())


loop.create_task(foo())
loop.run_forever()
```

### Use asyncio to manage multiprocess loops

This is a second and completely different way to set up two concurrent loops.
This is much less like browser JavaScript and more like Node which actually
uses multiprocesses in the background for IO. It's a bit like using a webworker.

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

loop = asyncio.get_event_loop()
# Change default from a ThreadPoolExecutor
loop.set_default_executor(ProcessPoolExecutor(2))

# Wrap the ugly asyncio API
def runAsyncSubprocess(loop, func, *args):
    return asyncio.ensure_future(loop.run_in_executor(None, func, *args))

# Note this is not async
def fooLoop(n):
    while True:
        foo(n)


# Also not async
def foo(n):
    print('foo %d' % n)


if __name__ == '__main__':
    runAsyncSubprocess(fooLoop, 1)
    runAsyncSubprocess(fooLoop, 2)
    loop.run_forever()
```

