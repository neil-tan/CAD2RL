from ..common.eventloop_pool import EventLoopPool
import asyncio
from time import sleep

def test_loop_start_end():
  pool = EventLoopPool(num_workers=4)
  loops = pool.loops
  assert len(pool.loops) == 4
  for loop in pool.loops:
    assert loop.is_running()
  del pool

def test_submit():
  pool = EventLoopPool(num_workers=4)
  coros = [asyncio.sleep(1) for _ in range(8)]
  results = pool.submit(coros)
  assert len(results) == 8
  for result in results:
    assert result is None
  del pool

def test_submit_with_results():
  pool = EventLoopPool(num_workers=4)
  coros = [asyncio.sleep(1, result=i) for i in range(8)]
  results = pool.submit(coros)
  assert len(results) == 8
  for i, result in enumerate(results):
    assert result == i
  del pool
