"""
File: callbacks.py
Author: Viet Nguyen

Description: This contains episode callbacks for every episode/step
"""

from typing import Dict
import numpy as np
import re
import io
from datetime import datetime
import jax

import embodied

def add_trajectory(ep, replay, positive_replay, worker):
  T = len(ep["is_first"])
  if ep['has_succeeded'][-1]:
    pos_ep = {**ep, "is_positive_memory": np.ones((T,))}
    [positive_replay.add({k: v[i] for k, v in pos_ep.items() if not k.startswith("log_")}) for i in range(list(pos_ep.values())[0].shape[0])]
    # print("episode successful, added to success buffer")
  else:
    # print("episdoe failed, did not add to success buffer")
    pass
  # the normal replay buffer has both positive and negative memories anyway
  norm_ep = {**ep, "is_positive_memory": np.ones((T,)) * ep['has_succeeded'][-1]}
  [replay.add({k: v[i] for k, v in norm_ep.items() if not k.startswith("log_")}, worker=worker) for i in range(list(norm_ep.values())[0].shape[0])]
  return ep

def lookback_on_episode(data: Dict[str, np.ndarray], args: embodied.Config) -> Dict[str, np.ndarray]:
  # The Prior Self-Revision Mechanism per episode
  # data: {k: (T, ...)}
  self_imitation_discount = args.self_imitation.disc
  self_imitation_discount_lower_threshold = args.self_imitation.thr
  failure_decay = args.self_imitation.fail
  T = len(data["is_first"])
  data["can_self_imitate"] *= data["has_succeeded"][-1]
  carry = np.maximum(data["can_self_imitate"][-1], 0.0)
  imitation_mask = [carry]
  for t in range(T - 1, 0, -1):
    carry = np.maximum(carry * self_imitation_discount, data["is_success"][t])
    carry *= (carry >= self_imitation_discount_lower_threshold).astype(np.float32)
    imitation_mask.insert(0, carry)
  imitation_mask = np.stack(imitation_mask) # (T)
  final_imitation_mask = np.clip(data["can_self_imitate"] + imitation_mask, 0, 1)
  positive_mask = np.concatenate([
    data["has_succeeded"][:1],
    final_imitation_mask[:-1]
  ], 0)
  failure_decays = np.asarray([failure_decay for i in range(T)]).cumprod(0)[::-1] / failure_decay
  failure_decays *= (1 - data["has_succeeded"][-1])
  neg_rates = -failure_decays
  neg_rates = neg_rates * (1 - data["can_self_imitate"])
  data["can_self_imitate"] = final_imitation_mask
  data["pref_label"] = positive_mask + neg_rates
  data["pref_label"][0] = 0
  data["pref_label"] = data["pref_label"].clip(-1, 1)
  return data


def print_episode(ep):
  length = len(ep['reward']) - 1
  score = float(ep['reward'].astype(np.float64).sum())
  print(f'Episode has {length} steps and return {score:.2f}.')
  return ep


#### Record each key value pair of the episode after every episode
def per_episode(ep, mode, nonzeros, args, logger, metrics):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    ep_stats = {
      'length': length,
      'score': score
    }
    if "relative_stability" in ep:
      ep_stats["relative_stability"] = ep["relative_stability"][-1]
    if "success_rate" in ep:
      ep_stats["success_rate"] = ep["success_rate"][-1]
    if "coverage_mean" in ep:
      ep_stats["coverage_mean"] = ep["coverage_mean"][-1]
    # record the number of steps that the agent is in the sucessful states in the episode.
    ep_stats["successfulness"] = np.sum(ep["is_success"].astype(np.float64)) / length
    logger.add(ep_stats, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    # print(f'Episode has {length} steps and return {score:.2f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix=('stats' if mode == 'train' else f'{mode}_stats'))
    return ep


def save(ep, epsdir):
  time = datetime.now().strftime("%Y%m%dT%H%M%S")
  uuid = str(embodied.uuid())
  score = str(np.round(ep['reward'].sum(), 1)).replace('-', 'm')
  length = len(ep['reward'])
  filename = epsdir / f'{time}-{uuid}-len{length}-rew{score}.npz'
  with io.BytesIO() as stream:
    np.savez_compressed(stream, **ep)
    stream.seek(0)
    filename.write(stream.read(), mode='wb')
  print('Saved episode:', filename)
  return ep


