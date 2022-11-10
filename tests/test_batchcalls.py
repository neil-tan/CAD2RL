from ..common.eventloop_pool import AsyncBatchedForward, new_thread_loop
import torch
import asyncio
from wrapt import synchronized

def forward(input):
  output = torch.zeros_like(input)
  for i in range(input.shape[0]):
    output[i] = input[i] * 2
  return output

@synchronized
def batch_forward_caller(batched_input):
  tensors = []
  for call_args in batched_input:
    args, kwargs = call_args
    tensor = args[0].squeeze(0)
    tensors.append(tensor)
  tensors = torch.stack(tensors, dim=0)
  return forward(tensors)

def get_dummy_samples():
  x = torch.rand(4, 1)
  y = forward(x)
  return x, y

def test_async_batched_forward():
  test_loop = new_thread_loop()[0]

  batched_forward = AsyncBatchedForward(batch_forward_caller, batch_size=2, timeout=0)

  x, y = get_dummy_samples()
  y_hat = torch.zeros_like(y)
  futures = []
  for i in range(x.shape[0]):
    futures.append(asyncio.run_coroutine_threadsafe(batched_forward(x[i].unsqueeze(0)), loop=test_loop))
  
  for i, future in enumerate(futures):
    y_hat[i] = future.result()

  assert torch.allclose(y, y_hat)