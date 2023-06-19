---
title: "Concurrency in Python - A Guide"
date: 2023-04-05
tags: [python, concurrency]
header:
  # image: "/images/kayla-koss-WSbd7BELXak-unsplash.jpg"
excerpt: "Fundamentals of concurrency in Python"
---
## Introduction
Concurrency is a key concept in programming. Python, a versatile and powerful language, provides various mechanisms to enable concurrent execution of tasks, unlocking the potential for improved efficiency and responsiveness in your applications. In this blog, we will explore the fundamentals of concurrency in Python and delve into different techniques and libraries that can help you leverage the power of parallelism.

## Understanding Concurrency

Concurrency refers to the ability of a program to execute multiple tasks in overlapping time periods. It enables different parts of a program to make progress simultaneously, improving overall efficiency and responsiveness. In concurrent programming, tasks can be executed independently or in parallel, depending on the programming model and underlying mechanisms used. It is essential in two scenarios:

**I/O-Bound Tasks**: Applications that involve I/O operations, such as reading from or writing to files, making network requests, or interacting with databases, can benefit from concurrency. While waiting for I/O to complete, other tasks can be executed, maximizing resource utilization.

```python
import requests
import time

def fetch_url(url):
    response = requests.get(url)
    print(f"Fetched {url}: {response.status_code}")

def main():
    urls = [
        "https://www.example.com",
        "https://www.openai.com",
        "https://www.python.org"
    ]

    start_time = time.time()

    # Fetch each URL sequentially
    for url in urls:
        fetch_url(url)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()
```
```
Total elapsed time: 1.0631120204925537 seconds
```

**CPU-Bound Tasks**: Computations that require significant processing power can be executed concurrently to leverage the capabilities of multi-core processors.
Breaking down a task into smaller subtasks and executing them in parallel can result in faster execution times.

```python
import time

def perform_calculation(num):
    result = 0
    for _ in range(num):
        result += 1
    return result

def main():
    # Number of CPU-bound tasks to execute
    num_tasks = 4

    start_time = time.time()

    # Perform the calculation sequentially
    results = [perform_calculation(10**7) for _ in range(num_tasks)]

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()
```
```
Total elapsed time: 1.1298840045928955 seconds
```

## Threading

Threading is a programming technique that allows concurrent execution of multiple threads within a single process.
When we execute I/O operations, by running these tasks in separate threads, other threads can continue their execution, preventing the entire program from being blocked.

```python
import threading
import requests
import time

def fetch_url(url):
    response = requests.get(url)
    print(f"Fetched {url}: {response.status_code}")

def main():
    urls = [
        "https://www.example.com",
        "https://www.openai.com",
        "https://www.python.org"
    ]

    start_time = time.time()

    # Create a list of threads to fetch each URL concurrently
    threads = [threading.Thread(target=fetch_url, args=(url,)) for url in urls]

    # Start the threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()
```
```
Total elapsed time: 0.7579257488250732 seconds
```

However there is a caveat to threading in python, known as the [GIL - Global Interpreter Lock](https://wiki.python.org/moin/GlobalInterpreterLock).
This is a mechanism that ensures only one thread executes Python bytecode at a time, preventing true parallel execution of threads and limiting the potential performance gains in CPU-bound multithreaded programs.

## Asynchronous Programming
In traditional synchronous programming, when a task is initiated, the program blocks and waits until that task completes before moving on to the next one.
This can result in inefficient resource utilization and reduced performance, especially when dealing with tasks that involve waiting for external resources.

Asynchronous programming, on the other hand, allows tasks to be initiated and executed independently, without waiting for each task to complete.
Instead of blocking, the program can continue with other tasks or operations while waiting for a response or completion of an asynchronous task.
This non-blocking behavior is achieved through the use of asynchronous functions, coroutines, event loops, and other mechanisms provided by asynchronous frameworks and libraries.

```python
import asyncio
import httpx
import time

async def fetch_url(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    print(f"Fetched {url}: {response.status_code}")

async def main():
    urls = [
        "https://www.example.com",
        "https://www.openai.com",
        "https://www.python.org"
    ]

    start_time = time.time()

    # Create a list of coroutines to fetch each URL concurrently
    coroutines = [fetch_url(url) for url in urls]

    # Execute the coroutines concurrently using asyncio
    await asyncio.gather(*coroutines)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    asyncio.run(main())
```
```
Total elapsed time: 0.7761232852935791 seconds
```

As a relatively minor drawback, you may notice we've used a different library httpx. The async implementation of python often relies on us having access to libraries with async functionality.

## Multiprocessing

While we are talking about Concurrency, a distinction should be made between it and Parallelism.

Parallelism is when two or more tasks are independently running at the same time. It might be helpful to think of concurrency as a property of a program, while Parallelism as a run-time behaviour.
In python this run-time behaviour is represented as each process running in its own Python interpreter and allows us to utilse more than one CPU core, as we are no longer restricted by the GIL.

Python achieves Parallelism through the `multiprocessing` library. Below we improve our CPU Bound task using this feature:

```python
import multiprocessing
import time

def perform_calculation(num):
    result = 0
    for _ in range(num):
        result += 1
    return result

def main():
    # Number of CPU-bound tasks to execute
    num_tasks = 4

    start_time = time.time()

    # Create a pool of worker processes
    pool = multiprocessing.Pool()

    # Perform the calculation concurrently using multiple processes
    results = pool.map(perform_calculation, [10**7] * num_tasks)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()
```
```
Total elapsed time: 0.7241818904876709 seconds
```

There a a great range of multiprocessing libraries available in Python, one great option is [Ray](https://www.ray.io/).

## Best Practices for Concurrency in Python

RealPython offers a great summary of what we have talked about so far:

| Concurrency Type                     | Switching Decision                                                    | Number of Processors |
| ------------------------------------ | --------------------------------------------------------------------- | -------------------- |
| Pre-emptive multitasking (threading) | The operating system decides when to switch tasks external to Python. | 1                    |
| Cooperative multitasking (asyncio)   | The tasks decide when to give up control.                             | 1                    |
| Multiprocessing (multiprocessing)    | The processes all run at the same time on different processors.       | Many                 |

So, how do we choose the right Concurrency Approach for your use case? In general the following applies:

| Type             | Concurrency Method  |
| ---------------- | ------------------- |
| CPU Bound        | Multiprocessing     |
| Blocking I/O     | asyncio / threading |
| Non-blocking I/O | asyncio             |

CPU tasks can also benefit from a variety of alternative mechanisms such as:
- GPU (Graphics Processing Unit) Computing such as [CUDA Python](https://developer.nvidia.com/cuda-python)
- Distributed Computing. In python we can use [Celery](). Outside of python we can use a distributed platform such as [Kubernetes](https://kubernetes.io/docs/tasks/job/fine-parallel-processing-work-queue/)

## Conclusion

Concurrency introduces new challenges and complexities, such as synchronization and resource management.
It's important to follow best practices, handle potential issues, and thoroughly test your code to ensure correctness and reliability.
With a solid understanding of concurrency and the right tools at your disposal, you can unlock the full potential of Python and your resources.

**_NOTE:_**  ChatGPT has been used to help generate some of the code snippets in this blog