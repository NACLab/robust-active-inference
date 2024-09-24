"""
File: agents.py
Author: Viet Nguyen
Date: 2024-04-05

Description: This contain the active inference agent that we are going to build
  to solve different robotic tasks.
"""

from typing import List, Dict, Tuple
import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from tensorflow_probability.substrates import jax as tfp
f32 = jnp.float32
tfd = tfp.distributions

import jaxagent
import jaxutils
import nets
import ninjax as nj
import epistemics


@jaxagent.Wrapper
class Agent(nj.Module):
  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(obs_space, act_space, config, name='wm')
    self.planner = ActiveInferencePlanner(
        self.wm, self.act_space, self.config, name='planner')

  def policy_initial(self, batch_size):
    return (
      self.wm.initial(batch_size),
      self.planner.initial(batch_size)
    )

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state = state
    embed = self.wm.encoder(obs)
    latent, _ = self.wm.rssm.obs_step(
        prev_latent, prev_action, embed, obs['is_first'])
    task_outs, task_state = self.planner.policy(latent, task_state)
    if mode == 'eval':
      outs = task_outs
      outs['action'] = outs['action'].sample(seed=nj.rng())
      outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
    elif mode == 'train':
      outs = task_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    state = ((latent, outs['action']), task_state)
    return outs, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    data = self.populate_data(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    # NOTE: train the AIF planner
    # state: prev_latent, prev_action
    context = {**data, **wm_outs['post']}  # (B, T, ...). wm_outs: embed, post, prior, priorpref
    C_context = {**data, **wm_outs['C_post']}
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    C_start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), C_context)
    _, _, mets = self.planner.train(self.wm.imagine, self.wm.imagine_preference, start, C_start, context)
    metrics.update(mets)
    outs = {}
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    data = self.populate_data(data)
    report = {}
    report.update(self.wm.report(data))
    mets = self.planner.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    return report

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs

  def populate_data(self, data: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
    # Just for populating data when initialize params
    B, T, *A = data["action"].shape
    if "can_self_imitate" not in data:
      data["can_self_imitate"] = jnp.zeros((B, T), jnp.float32)
    if "pref_label" not in data:
      data["pref_label"] = jnp.zeros((B, T), jnp.float32)
    if "unrelated_action" not in data:
      data["unrelated_action"] = jnp.zeros((B, T, *A), jnp.float32)
    return data


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}

    # WM Modules
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    self.rssm = nets.CRSPP_RSSM(**config.rssm, name='rssm')
    self.heads = {
      'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
      'reward': nets.MLP((), **config.reward_head, name='rew'),
      'cont': nets.MLP((), **config.cont_head, name='cont'),
    }
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)

    # Information gain module here
    self.epistemic_module = epistemics.InformationGain(config, name="epis")

    # Other chores
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def cosine_sim(self, s1, s2):
    if self.rssm._classes:
      shape = s1['stoch'].shape[:-2] + (self.rssm._stoch * self.rssm._classes,)
      x1 = s1['stoch'].reshape(shape)
      x2 = s2['stoch'].reshape(shape)
    else:
      x1 = s1['stoch']
      x2 = s2['stoch']
    x1 = jnp.concatenate([s1['deter'], x1], -1)
    x2 = jnp.concatenate([s2['deter'], x2], -1)
    # x1 and x2 (B, T, L)
    dot = (x1 * x2).sum(-1)
    denom_x1 = jnp.sqrt((x1 ** 2).sum(-1).clip(1e-8))
    denom_x2 = jnp.sqrt((x2 ** 2).sum(-1).clip(1e-8))
    return dot / (denom_x1 * denom_x2).clip(1e-8)

  def train(self, data, state):
    modules = [self.encoder, self.rssm, *self.heads.values()]
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    epis_mets = self.epistemic_module.train({**data, **outs['post']}) # {deter: (B, T, ...), stoch: (B, T, ...), 'action': (B, T, ...), ...}
    metrics.update(epis_mets)
    return state, outs, metrics

  def loss(self, data, state):
    embed = self.encoder(data)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, prior = self.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    C_post, C_prior = self.rssm.observe_no_action(
        embed, data['is_first'], prev_latent)

    # Normal observation inference
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      # Head inference from RSSM
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)

    #### COMPUTE LOSSES
    # compute loss for normal world model
    losses = {}
    losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
    for key, dist in dists.items():
      loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())

    # computing loss for prior prefernece model
    C_losses = {}
    C_losses['dyn'] = self.rssm.dyn_loss(C_post, C_prior, **self.config.dyn_loss)
    C_losses['rep'] = self.rssm.rep_loss(C_post, C_prior, **self.config.rep_loss)
    positive_kl_mask = jnp.sign(data['pref_label']).clip(0, 1)
    C_scales = {**self.scales, 'dyn': positive_kl_mask * self.scales['dyn'], 'rep': positive_kl_mask * self.scales['rep']}
    C_scaled = {k: v * C_scales[k] for k, v in C_losses.items()}
    C_model_loss = sum(C_scaled.values())
    similarity = self.cosine_sim(C_post, sg(post)) # (B, T)
    C_losses["contrast"] = -data['pref_label'] * similarity
    C_model_loss_mean = C_model_loss.mean() + self.scales['contrast'] * C_losses["contrast"].mean() # because contrastive loss is not the same shape as the other element

    out = {'embed':  embed, 'post': post, 'prior': prior, 'C_post': C_post}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    out.update({f'{k}_C_loss': v for k, v in C_losses.items()})
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = data['action'][:, -1]
    state = last_latent, last_action
    metrics = self._metrics(data, dists, post, prior, C_post, C_prior,
      losses, C_losses, model_loss, C_model_loss, similarity)
    return model_loss.mean() + C_model_loss_mean, (state, out, metrics)

  def imagine(self, policy, start, horizon):
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    keys = list(self.rssm.initial(1).keys())
    start = {k: v for k, v in start.items() if k in keys}
    start['action'] = policy(start)
    def step(prev, _):
      prev = prev.copy()
      state = self.rssm.img_step(prev, prev.pop('action'))
      return {**state, 'action': policy(state)}
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    traj = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
    cont = self.heads['cont'](traj).mode()
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def imagine_preference(self, start, horizon):
    # With current state at the first position
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    keys = list(self.rssm.initial_all_zeros(1).keys())
    start = {k: v for k, v in start.items() if k in keys}
    def step(prev, _):
      prev = prev.copy()
      state = self.rssm.img_step_no_action(prev)
      return state
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    traj = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
    cont = self.heads['cont'](traj).mode()
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def crspp_traj_sim(self, traj, pref_traj):
    sim = (self.cosine_sim(traj, sg(pref_traj))) # (H, BT)
    return sim

  def report(self, data):
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])

    ### Normal rssm imagine
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)

    ## Prior preference imagine
    C_context, _ = self.rssm.observe_no_action(
        self.encoder(data)[:6, :5], data['is_first'][:6, :5])
    C_start = {k: v[:, -1] for k, v in C_context.items()}
    C_recon = self.heads['decoder'](C_context)
    C_openl = self.heads['decoder'](
        self.rssm.imagine_no_action(6, self.config.batch_length - 5, C_start))
    # C_openl = self.heads['decoder'](
    #     self.rssm.imagine_no_action(6, self.config.batch_length - 5, start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      if key in self.config.run.log_keys_video:
        truth = data[key][:6].astype(jnp.float32)
        # model = jnp.concatenate([recon[key].mode()[:, :5], C_openl[key].mode()], 1)
        model = jnp.concatenate([C_recon[key].mode()[:, :5], C_openl[key].mode()], 1)
        error = (model - truth + 1) / 2
        video = jnp.concatenate([truth, model, error], 2)
        report[f'openl_pref_{key}'] = jaxutils.video_grid(video)

    return report

  def _metrics(self, data, dists, post, prior, C_post, C_prior,
      losses, C_losses, model_loss, C_model_loss, similarity):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics.update({f'C_{k}_loss_mean': v.mean() for k, v in C_losses.items()})
    metrics.update({f'C_{k}_loss_std': v.std() for k, v in C_losses.items()})
    metrics.update(jaxutils.tensorstats(entropy(C_prior), 'C_prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(C_post), 'C_post_ent'))
    metrics.update(jaxutils.tensorstats(similarity, 'similarity'))
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['C_model_loss_mean'] = C_model_loss.mean()
    metrics['C_model_loss_std'] = C_model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    return metrics


class ActiveInferencePlanner(nj.Module):

  def __init__(self, wm: WorldModel, act_space, config):
    rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
    crsppfn = lambda s, p: wm.crspp_traj_sim(s, p)[1:]
    igfn = lambda s: wm.epistemic_module(s)[1:]
    fns = [rewfn, crsppfn, igfn]

    critics = {'extr': GFunction(fns, config, name='critic')}
    self.ac = ImagActorCritic(
        critics, {'extr': 1.0}, act_space, config, name='ac')

  def initial(self, batch_size):
    return self.ac.initial(batch_size)

  def policy(self, latent, state):
    return self.ac.policy(latent, state)

  def train(self, imagine, imagine_preference, start, C_start, data):
    return self.ac.train(imagine, imagine_preference, start, C_start, data)

  def report(self, data):
    return {}


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def get_dist(self, state, argmax=False):
    if self.config.rssm.classes:
      logit = state['logit'].astype(f32)
      return tfd.Independent(jaxutils.OneHotDist(logit), 1)
    else:
      mean = state['mean'].astype(f32)
      std = state['std'].astype(f32)
      return tfd.MultivariateNormalDiag(mean, std)

  def train(self, imagine, imagine_preference, start, C_start, context):
    """

    Args:
        imagine: (Callable): world model imagine function
        start (Dict[str, jnp.ndarray]): start = {**data, **wm_outs} # (B*T, ...). (same with context but flattened batch and time)
        context (Dict[str, jnp.ndarray]): context = {**data, **wm_outs} # (B, T, ...).
          wm_outs: embed, post, prior, prev_latents, priorpref

    Returns:
      traj: rollout states given policy{stoch, deter, mean, std} (B, T, ...)
      metrics: the metrics
    """
    # Actor training, maximizing instrumental
    # train the planner that maximizing the instrumental signal:
    # value + prior preference value + actor refresh
    # minimizing epistemic signal => maximizing parameter information gain
    # maximizing state information gain
    def loss(start, C_start):
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      pref_traj = imagine_preference(C_start, self.config.imag_horizon)
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss(traj, pref_traj, start)
      return loss, (traj, pref_traj, metrics)
    mets, (traj, pref_traj, metrics) = self.opt(self.actor, loss, start, C_start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, pref_traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, pref_traj, metrics

  def loss(self, traj, pref_traj, start):
    metrics = {}

    # Train actor that maximizing instrumental and epistemic values
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      G, ret, base, G_mets = critic.score(traj, pref_traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(G, f'{key}_G'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      for k, v in G_mets.items():
        metrics.update(jaxutils.tensorstats(v, f'{key}_{k}'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)

    # Actor entropy loss, minimizing entropy of the actor
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent # NOTE: one can maximize the action/policy uncertainty to boost exploration
    loss *= sg(traj['weight'])[:-1]

    # entropy of the state from taking action
    wment = self.get_dist(traj).entropy()[1:] # (H-1, BT, ...): s_t, (a_t) => (s_t+1)
    loss -= self.config.wment * wment # NOTE: one can maximize the generative world model uncertainty to boost exploration. If we minimize wment instead, there is an effect that reduce agent's cumulative reward trend in mountain car w/o expert

    # Actor refresh loss
    policy2 = self.actor(sg(start))
    logpi2 = policy2.log_prob(sg(start['action']))
    actor_refresh = -logpi2 * sg(start['can_self_imitate']) # minimize this piece to do self imitation learning
    actor_refresh *= self.config.loss_scales.actor_refresh

    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv, actor_refresh, wment)) # TODO: logging also the actor refresh term for metrics
    return loss.mean() + actor_refresh.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv, actor_refresh, wment):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(wment, 'policy_wm_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics.update(jaxutils.tensorstats(actor_refresh, 'actor_refresh'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class GFunction(nj.Module):

  def __init__(self, fns, config):
    rewfn, crsppfn, igfn = fns
    self.rewfn = rewfn
    self.crsppfn = crsppfn
    self.igfn = igfn

    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, pref_traj, actor):
    target = sg(self.score(traj, pref_traj, actor)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, pref_traj, actor=None):
    """minimizing expected free energy also means:
      - minimizing parameter information gain
      - maximizing reward
    """
    # discounted cumulative negative expected free energy
    # at each step:
    # - instrumental (reward + similarity). Aim to increase reward and similarity
    # - epistemic (negative param info gain). Aim to decreace information gain
    rew = self.rewfn(traj)
    crsppscore = self.crsppfn(traj, pref_traj)
    mets = {'G_rew': rew, 'G_crspp': crsppscore}
    neg_param_info_gain = -self.igfn(traj)
    mets['G_ig'] = neg_param_info_gain
    G = self.config.G_scales.rew * rew +\
      self.config.G_scales.crspp * crsppscore + \
      self.config.G_scales.ig * neg_param_info_gain
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    assert len(G) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.net(traj).mean()
    vals = [value[-1]]
    interm = G + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return G, ret, value[:-1], mets
