r"""Example running contrastive RL in JAX.

Run using multi-threading
  python lp_contrastive.py --lp_launch_type=local_mt


"""
import functools
from typing import Any, Dict

from absl import app
from absl import flags
import contrastive
from contrastive import utils as contrastive_utils
import launchpad as lp
import numpy as np
import os
import wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir_path', 'logs/', 'Where to log metrics')
flags.DEFINE_integer('time_delta_minutes', 5, 'how often to save checkpoints')
flags.DEFINE_integer('seed', 42, 'Specify seed, only used if use_slurm_array is false')
flags.DEFINE_bool('add_uid', False, 'Whether to add a unique id to the log directory name')
flags.DEFINE_string('alg', 'contrastive_cpc', 'Algorithm type, e.g. default is contrastive_cpc with no entropy or KL losses')
flags.DEFINE_string('env', 'sawyer_bin', 'Environment type, e.g. default is sawyer bin')
flags.DEFINE_integer('num_steps', 8_000_000, 'Number of steps to run', lower_bound=0)
flags.DEFINE_bool('sample_goals', False, 'sample the goal position uniformly according to the environment (corresponds to the original contrastive_rl algorithm)')
flags.DEFINE_bool('render', False, 'Enable rendering during training (only works with local execution)')

# fixed goal coordinates for supported environments
fixed_goal_dict={
    'point_Spiral11x11': [np.array([5,5], dtype=float), np.array([10,10], dtype=float)],
    'sawyer_bin': np.array([0.12, 0.7, 0.02]),
    'sawyer_box': np.array([0.0, 0.75, 0.133]),
    'sawyer_peg': np.array([-0.3, 0.6, 0.0]),
    # Uncomment this to use manual fixed goal
    # 'stretch_pick': (
    #     np.zeros(153, dtype=float),  # start state (will be ignored if not used)
    #     np.concatenate([
    #         # Robot state (13 dims) - lifted position with object grasped
    #         np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0,  # base pose (centered, upright)
    #                   0.9, 0.2,                        # lift high, arm extended
    #                   0.0, 0.0, 0.0,                   # wrist neutral
    #                   -0.02,                           # gripper (fully closed) ← within valid range [-0.02, 0.04]
    #                   1.0]),                           # is_grasped=True
    #         # Object 0 state - now position-first
    #         np.array([0.5, 0.0, 0.85,              # position (x, y, z)
    #                   1.0, 0.0, 0.0, 0.0]),        # quaternion (qw, qx, qy, qz)
    #         # Objects 1-19 (133 dims) - all zeros (inactive)
    #         np.zeros(19 * 7, dtype=float)
    #     ])
    # )
}

@functools.lru_cache
# def get_env(env_name, start_index, end_index, seed, fix_goals = False, fix_goals_actor = False, use_naive_sampling=False, clock_period=None):
def get_env(env_name, start_index, end_index, seed, fix_goals = False, fix_goals_actor = False, use_naive_sampling=False, clock_period=None, render_mode=None):
  if fix_goals:
    fixed_start_end = fixed_goal_dict.get(env_name, None)
  else:
    fixed_start_end = None
    
  return contrastive_utils.make_environment(env_name, start_index, end_index, seed=seed, fixed_start_end = fixed_start_end, render_mode=render_mode)


def get_program(params):
  """Constructs the program."""

  env_name = params['env_name']
  seed = params['seed']
  
  # Extract render_mode before creating config (it's not a config parameter)
  render_mode = params.pop('render_mode', None)

  config = contrastive.ContrastiveConfig(**params)
  
  fix_goals = params['fix_goals']

  if fix_goals:
    fixed_start_end = fixed_goal_dict.get(env_name, None)
  else:
    fixed_start_end = None
    
  # Actor environment factory: NO rendering (to avoid GLFW threading conflicts)
  env_factory = lambda seed: contrastive_utils.make_environment(  # pylint: disable=g-long-lambda
      env_name, config.start_index, config.end_index, seed, fixed_start_end = fixed_start_end, render_mode = None)

  env_factory_no_extra = lambda seed: env_factory(seed)[0]  # Remove obs_dim.
    
  # Evaluator uses render_mode
  environment, obs_dim = get_env(env_name, config.start_index,
                                 config.end_index, seed, fix_goals = fix_goals, render_mode = render_mode)

  assert (environment.action_spec().minimum == -1).all()
  assert (environment.action_spec().maximum == 1).all()
  config.obs_dim = obs_dim
  config.max_episode_steps = getattr(environment, '_step_limit') + 1
  network_factory = functools.partial(
      contrastive.make_networks, obs_dim=obs_dim, repr_dim=config.repr_dim,
      repr_norm=config.repr_norm, twin_q=config.twin_q,
      use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes)
    
  env_factory_fixed_goals = lambda seed: contrastive_utils.make_environment(  # pylint: disable=g-long-lambda
      env_name, config.start_index, config.end_index, seed, fixed_start_end = fixed_goal_dict.get(env_name, None), render_mode = render_mode)
  env_factory_no_extra_fixed_goals = lambda seed: env_factory_fixed_goals(seed)[0]  # Remove obs_dim.
    
  agent = contrastive.DistributedContrastive(
      seed=seed,
      environment_factory=env_factory_no_extra,
      environment_factory_fixed_goals=env_factory_no_extra_fixed_goals,
      network_factory=network_factory,
      config=config,
      num_actors=config.num_actors,
      log_to_bigtable=True,
      max_number_of_steps=config.max_number_of_steps)
  return agent.build()


def main(_):
  # Create experiment description.

  # 1. Select an environment.
  # Supported environments:
  #   Metaworld: sawyer_{bin,box,peg}
  #   2D nav: point_{Spiral11x11}
  env_name = FLAGS.env
  print('Using env {}...'.format(env_name))
  
  seed_idx = FLAGS.seed
  print('Using random seed {}...'.format(seed_idx))
  params = {
      'seed': seed_idx,
      'use_random_actor': True,
      # entropy_coefficient = None will use adaptive; if setting to a number, note this is log alpha
      'entropy_coefficient': 0.0,
      'env_name': env_name,
      # the number of environment steps
      'max_number_of_steps': FLAGS.num_steps,
      'render_mode': 'human' if FLAGS.render else None,
      'num_actors': 1 if FLAGS.render else 4,  # Use only 1 actor when rendering
  }
  # 2. Select an algorithm. The currently-supported algorithms are:
  # contrastive_nce, contrastive_cpc, c_learning, nce+c_learning
  # Many other algorithms can be implemented by passing other parameters
  # or adding a few lines of code.
  # By default, do contrastive CPC
  alg = FLAGS.alg
  print('Using alg {}...'.format(alg))
  params['alg_name'] = alg
  params['fix_goals'] = not FLAGS.sample_goals
  add_uid = FLAGS.add_uid
  params['add_uid'] = add_uid
  print('Adding uid: {}...'.format(params['add_uid']))
  
  params['log_dir'] = FLAGS.log_dir_path
  params['time_delta_minutes'] = FLAGS.time_delta_minutes
  
  if alg == 'contrastive_cpc':
    params['use_cpc'] = True
  elif alg == 'c_learning':
    params['use_td'] = True
    params['twin_q'] = True
  elif alg == 'nce+c_learning':
    params['use_td'] = True
    params['twin_q'] = True
    params['add_mc_to_td'] = True
  else:
    raise NotImplementedError('Unknown method: %s' % alg)

  

  program = get_program(params)
  # Set terminal='tmux' if you want different components in different windows.
  
  print(params)
  
  lp.launch(program, terminal='current_terminal')

if __name__ == '__main__':
  app.run(main)
