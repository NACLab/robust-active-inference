"""
File: epistemics.py
Author: Viet Nguyen
Date: 2024-04-05

Description: This contains implementation for epistemic signals for the agent.
"""

import jax
import jax.numpy as jnp
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
f32 = jnp.float32

import nets
import jaxutils
import ninjax as nj


class InformationGain(nj.Module):

  def __init__(self, config):
    self.config = config.update({'ig_head.inputs': ['tensor']})
    self.opt = jaxutils.Optimizer(name='ig_opt', **config.ig_opt)
    self.inputs = nets.Input(config.ig_head.inputs, dims='deter')
    self.target = nets.Input(self.config.ig_target, dims='deter')
    self.nets = [nets.MLP(shape=None, **self.config.ig_head, name=f'ig{i}')
      for i in range(self.config.ig_models)]

  def _entropy(self, stddev: jax.Array) -> jax.Array:
    return 1.0 / 2 * jnp.log((2 * jnp.pi * stddev**2).clip(1e-8)) + 1.0 / 2

  def __call__(self, traj):
    inp = self.inputs(traj)
    dists = [net(inp) for net in self.nets]
    preds = jnp.asarray([dist.sample(seed=nj.rng()) for dist in dists])
    mixsigma = preds.std(0)
    ent_avg = self._entropy(mixsigma).mean(-1)
    avg_ent = jnp.asarray([self._entropy(dist.stddev()).mean(-1) for dist in dists]).mean(0)
    dynamic_ig_norm = 1.0 / jnp.sqrt(self.config.rssm.stoch)
    params_info_gain = (ent_avg - avg_ent) * dynamic_ig_norm
    return params_info_gain

  def train(self, data):
    return self.opt(self.nets, self.loss, data)

  def loss(self, data):
    inp = sg(self.inputs(data)[:, :-1])
    tar = sg(self.target(data)[:, 1:])
    losses = []
    for net in self.nets:
      net._shape = tar.shape[2:] # set the shape here
      loss = -net(inp).log_prob(tar).mean()
      losses.append(loss)
    probs = jax.random.uniform(nj.rng(), (len(losses),))
    nolearn = 1 - (probs < self.config.ig_head_random_no_learning).astype(jnp.float32)
    scales = nolearn.clip(1e-8)
    return (scales * jnp.array(losses)).sum()

