import asyncio
from threading import Thread
from typing import Iterable

class EventLoopPool:
  def __init__(self, num_workers:int=4):
    self.num_workers = num_workers
    self.loops = []
    for _ in range(num_workers):
      self.loops.append(self.new_thread_loop())
  
  def new_thread_loop(self, loop:asyncio.AbstractEventLoop=None):
    loop = asyncio.new_event_loop() if loop is None else loop
    def loop_thread():
      asyncio.set_event_loop(loop)
      loop.run_forever()

    thread = Thread(target=loop_thread, daemon=True)
    thread.start()
    return loop

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
      loop.call_soon_threadsafe(stop_loop(loop))
      loop.call_soon_threadsafe(loop.stop)
