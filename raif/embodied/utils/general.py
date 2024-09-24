import numpy as np

def tensorstats(tensor, prefix=None):
  metrics = {
      'mean': tensor.mean(),
      'std': tensor.std(),
      'mag': np.abs(tensor).max(),
      'min': tensor.min(),
      'max': tensor.max(),
      'dist': subsample(tensor),
  }
  if prefix:
    metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
  return metrics

def subsample(values, amount=512):
  values = values.flatten()
  if len(values) > amount:
    values = np.random.permutation(values)[:amount]
  return values