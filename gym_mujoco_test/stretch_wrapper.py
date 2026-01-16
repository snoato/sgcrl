import dm_env
from dm_env import specs
import numpy as np
from gym_mujoco_test.stretch_pick_env import StretchPickEnv

class StretchPickDMEnv(dm_env.Environment):
    """Wraps StretchPickEnv to match dm_env interface expected by contrastive RL."""
    
    MAX_OBJECTS = 20
    OBJ_FEATS = 7
    ROBOT_FEATS = 5  # base_x, lift, arm_ext, gripper, is_grasped (removed wrist yaw/pitch/roll)
    
    def __init__(self, seed=0, num_objects=1, objects=None, max_episode_steps=300, fixed_start_end=None, render_mode=None):
        """
        Args:
            num_objects: Number of objects on table (default=1 for Single Object curriculum stage)
                         Increase to 2, 5, etc. for later curriculum stages
            objects: Specific object IDs to use (e.g., [0, 5, 10])
            fixed_start_end: Tuple of (start_obs, goal_obs) for fixed goal setting
            render_mode: Rendering mode ('human' for window, None for no rendering)
        """
        print(f"\n{'='*60}")
        print(f"[StretchPickDMEnv] Creating environment")
        print(f"  render_mode: {render_mode}")
        print(f"  num_objects: {num_objects}")
        print(f"  objects: {objects}")
        print(f"{'='*60}\n")
        self._env = StretchPickEnv(num_objects=num_objects, objects=objects, fixed_start_end=fixed_start_end, render_mode=render_mode)
        self._seed = seed
        self._reset_next_step = True
        self._step_limit = max_episode_steps if max_episode_steps else 500
        self._step_count = 0
        self._fixed_start_end = fixed_start_end
        
        # Get the original action space bounds for rescaling
        self._action_low = self._env.action_space.low
        self._action_high = self._env.action_space.high
        self._action_dim = self._env.action_space.shape[0]
        
        # Observation dimension: 5 robot + 20*7 objects = 145
        # Full observation is [obs, goal] so 145 * 2 = 290
        self._obs_dim = self.ROBOT_FEATS + self.MAX_OBJECTS * self.OBJ_FEATS
        self._debug_print_obs = True 
        
    # def _rescale_action(self, action):
    #     """Rescale action from [-1, 1] to environment's action space."""
    #     action = np.clip(action, -1.0, 1.0)
    #     scaled_action = self._action_low + (action + 1.0) * 0.5 * (self._action_high - self._action_low)
    #     return scaled_action
        
    def _rescale_action(self, action):
        """Option B: no rescaling here. Env expects [-1, 1] directly."""
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def reset(self):
        self._reset_next_step = False
        self._step_count = 0
        obs, _ = self._env.reset()
        
        # Environment now returns [obs, goal] concatenated
        # Ensure it's a 1D array
        full_obs = np.asarray(obs, dtype=np.float32).flatten()

        # ---- DEBUG PRINT ----
        if not hasattr(self, "_did_print_obs"):
            print("full_obs shape:", full_obs.shape)
            print("Expected: 290 (145 obs + 145 goal)")
            print("first 5 dims (robot state):", full_obs[:5])
            print("object part (first object):", full_obs[5:12])
            print("goal part (first 5 dims):", full_obs[145:150])
            self._did_print_obs = True
        # -----------------------------------
        
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=full_obs
        )
    
    def step(self, action):
        if self._reset_next_step:
            return self.reset()

        #================================================================
        a_in = np.asarray(action)  # raw from policy (likely jax -> numpy already, but safe)

        if not hasattr(self, "_dbg_step"):
            self._dbg_step = 0
        self._dbg_step += 1

        if self._dbg_step % 20 == 0:
            print("[POLICY->DMENV] action in:",
                np.round(a_in[:], 3),
                "shape", a_in.shape,
                "min/max", float(a_in.min()), float(a_in.max()))
        #================================================================
        
        # Rescale action from [-1, 1] to environment's action space
        scaled_action = self._rescale_action(action)

        #================================================================
        a_out = np.asarray(scaled_action)

        if self._dbg_step % 20 == 0:
            print("[DMENV] action out:",
                np.round(a_out[:], 3),
                "min/max", float(a_out.min()), float(a_out.max()))
        #================================================================
        
        self._step_count += 1
        obs, reward, terminated, truncated, info = self._env.step(scaled_action)
        
        # Environment returns [obs, goal] concatenated
        # Ensure it's a 1D array
        full_obs = np.asarray(obs, dtype=np.float32).flatten()
        
        # Check if episode should end
        done = terminated or truncated or (self._step_count >= self._step_limit)
        
        if done:
            self._reset_next_step = True
            step_type = dm_env.StepType.LAST
            discount = 0.0
        else:
            step_type = dm_env.StepType.MID
            discount = 1.0
            
        return dm_env.TimeStep(
            step_type=step_type,
            reward=float(reward),
            discount=discount,
            observation=full_obs
        )
    
    def observation_spec(self):
        # Observation is [current_state, goal_state] concatenated
        full_obs_dim = self._obs_dim * 2  # 148 * 2 = 296
        
        obs_low = np.full(full_obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(full_obs_dim, np.inf, dtype=np.float32)
        
        return specs.BoundedArray(
            shape=(full_obs_dim,), 
            dtype=np.float32, 
            minimum=obs_low,
            maximum=obs_high,
            name='observation'
        )
    
    def action_spec(self):
        # Contrastive RL expects actions in [-1, 1]
        return specs.BoundedArray(
            shape=(self._action_dim,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='action'
        )
    
    def close(self):
        self._env.close()