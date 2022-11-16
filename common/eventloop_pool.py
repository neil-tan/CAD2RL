import asyncio
from threading import Thread
from typing import Iterable
from collections import namedtuple
import time
import torch

def new_thread_loop(loop:asyncio.AbstractEventLoop=None):
  loop = asyncio.new_event_loop() if loop is None else loop
  def loop_thread():
    asyncio.set_event_loop(loop)
    loop.run_forever()

  thread = Thread(target=loop_thread, daemon=True)
  thread.start()
  return loop, thread

class EventLoopPool:
  def __init__(self, num_workers:int=4):
    self.num_workers = num_workers
    self.loops = []
    for _ in range(num_workers):
      loop, _thread = new_thread_loop()
      self.loops.append(loop)
  

  def submit(self, coros:Iterable[asyncio.coroutine]):
    l = len(coros)
    batch = l // self.num_workers
    remainder = l % self.num_workers
    futures = []

    for i, loop in enumerate(self.loops):
      start = i * batch
      end = start + batch
      if i == self.num_workers - 1:
       end += remainder

      for coro in coros[start:end]:
        futures.append(asyncio.run_coroutine_threadsafe(coro, loop=loop))

    results = [future.result() for future in futures]

    return results
  
  def __del__(self):
    def stop_loop(loop):
      loop.stop()
    
    for loop in self.loops:
      loop.call_soon_threadsafe(stop_loop, loop)
      loop.call_soon_threadsafe(loop.stop)


class AsyncBatchedForward:
  def __init__(self, batched_func:callable, batch_size:int=0, timeout:int=0, loop:asyncio.AbstractEventLoop=None):
    self.loop = asyncio.get_event_loop() if loop is None else loop
    self.batched_func = batched_func
    self.timeout = timeout

    self.batch_size = batch_size

    self.time_last_flush = 0
    self.call_stack_entry = namedtuple('call_stack_entry', ['future', 'args', 'kwargs']) # (future, (args, kwargs))
    self.call_stack = []

  def __should_flush(self):
    if self.batch_size > 0 and len(self.call_stack) >= self.batch_size:
      return True

    if self.timeout > 0 and time.time() - self.time_last_flush >= self.timeout:
      return True

    return False
  
  async def __flush(self):
    futures = []
    inputs = []
    
    self.time_last_flush = time.time()
    for entry in self.call_stack:
      futures.append(entry.future)
      inputs.append((entry.args, entry.kwargs))

    outputs = self.batched_func(inputs)

    for future, output in zip(futures, outputs):
      future.set_result(output)
    
    self.call_stack.clear()

  async def __batch_call(self, *args, **kwargs):
    # assume that we are thread safe here
    future = asyncio.Future()
    self.call_stack.append(self.call_stack_entry(future, args, kwargs))

    if self.__should_flush():
      await self.__flush()
    
    result = await future
    return result

  def flush(self):
    self.loop.call_soon_threadsafe(self.__flush)

  async def __call__(self, *args, **kwargs):
    current_loop = asyncio.get_running_loop()
    future = current_loop.create_task(self.__batch_call(*args, **kwargs))
    return await future
    