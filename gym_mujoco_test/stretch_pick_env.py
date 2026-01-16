import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box
from .mujoco_env import MujocoEnv
import mujoco
import copy
from .ik.mink_ik import MinkIK



#----------------------
# Base joint: qpos[0] = [x] (only x-axis slide joint)
IDX_BASE_X      = 0  # base_x (slide joint) - ONLY degree of freedom for base

# Robot joints (shifted down by 6 from freejoint version)
IDX_RIGHT_WHEEL = 1
IDX_LEFT_WHEEL  = 2
IDX_LIFT        = 3                 # joint_lift
IDX_ARM_L3      = 4                 # joint_arm_l3 (telescope segment)
IDX_ARM_L2      = 5                 # joint_arm_l2 (telescope segment)
IDX_ARM_L1      = 6                 # joint_arm_l1 (telescope segment)
IDX_ARM_L0      = 7                 # joint_arm_l0 (telescope segment)
IDX_WRIST_YAW   = 8                 # joint_wrist_yaw
IDX_WRIST_PITCH = 9                 # joint_wrist_pitch
IDX_WRIST_ROLL  = 10                # joint_wrist_roll
IDX_GRIPPER     = 11                # joint_gripper_slide
IDX_GRIPPER_LEFT = 12               # joint_gripper_finger_left_open
IDX_RUBBER_LEFT_X = 13
IDX_RUBBER_LEFT_Y = 14
IDX_GRIPPER_RIGHT = 15              # joint_gripper_finger_right_open
IDX_RUBBER_RIGHT_X = 16
IDX_RUBBER_RIGHT_Y = 17
IDX_HEAD_PAN    = 18
IDX_HEAD_TILT   = 19
IDX_HEAD_NAV    = 20

# Object freejoint: qpos[21:28] (shifted down by 6 from original 27)
IDX_OBJ_X       = 21                # object0:joint x
IDX_OBJ_Y       = 22                # object0:joint y
IDX_OBJ_Z       = 23                # object0:joint z
IDX_OBJ_QW      = 24
IDX_OBJ_QX      = 25
IDX_OBJ_QY      = 26
IDX_OBJ_QZ      = 27  
#----------------------

class StretchPickEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }

    MAX_OBJECTS = 20
    OBJ_FEATS = 7   # x, y, z, qw, qx, qy, qz
    ROBOT_FEATS = 5  # base_x, lift, arm_ext, gripper, is_grasped (removed wrist_yaw, wrist_pitch, wrist_roll)

    def __init__(self, num_objects=None, objects=None, fixed_start_end=None, **kwargs):

        if objects is not None:
        #ex:[10, 15]
            self.objects = list(objects)
            self.num_objects = len(self.objects)
        else:
        # if no number, then use 1
            if num_objects is None:
                num_objects = 1
            self.num_objects = num_objects
            self.objects = list(range(num_objects))
        
        print(f"[StretchPickEnv] Initializing with num_objects={self.num_objects}, objects={self.objects}")
        
        assert 1 <= self.num_objects <= self.MAX_OBJECTS, \
            f"num_objects must be 1-{self.MAX_OBJECTS}"
        assert max(self.objects) < self.MAX_OBJECTS, \
            f"object id {max(self.objects)} exceeds MAX_OBJECTS={self.MAX_OBJECTS}"
       
        self._fixed_start_end = fixed_start_end
        self._goal = None  # Will be set on first reset
        

        utils.EzPickle.__init__(self, num_objects=self.num_objects, **kwargs)

        # Fixed observation space for 20 objects
        # Observation is [current_state, goal_state] concatenated
        obs_dim = self.ROBOT_FEATS + self.MAX_OBJECTS * self.OBJ_FEATS
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim * 2,), dtype=np.float64)

        MujocoEnv.__init__(
            self, "simple_scene.xml", 50, observation_space=observation_space, **kwargs
        )
        
        # ---------------------------------------------------------------------------------------------------------------------
        print("========== ACTION SPACE DEBUG ==========")
        print("nu (number of actuators):", self.model.nu)
        print("Actuator names:")
        for i in range(self.model.nu):
            print(f"  {i}: {self.model.actuator(i).name}")
        print("\nCtrlrange:")
        print(self.model.actuator_ctrlrange)

        print("\nGym action_space:")
        print(self.action_space)
        print("========================================")

        #---------------------------------------------------------------------------------------------------------------------
        model = self.model                  # mujoco MjModel
        self.target_obj_id = 0              # Target object ID
        self.object_geom_ids = {}
        for obj_id in range(self.MAX_OBJECTS):
            name = f"object{obj_id}"
            try:
                self.object_geom_ids[obj_id] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                
            except Exception as e:
                print(f"Warning: Could not find geom names: {e}")
                self.object_geom_ids[obj_id] = None
                # self.finger_geom_ids = []

        self.finger_geom_ids = [
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_geom"),
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_geom"),
                ]

        self.ik = MinkIK(
            model=self.model,
            data=self.data,
            target_name="target0"
        )

        # self._lift_streak = 0            # (testing2)
        # self._lift_streak_required = 20   # K=20 = 1.0 sec consecutive steps to confirm lift (testing2)
        
        self._last_successful_ik_qpos = None

        # Print joint information, to check if the index is correct
        print("Number of joints:", model.njnt)
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_adr = model.jnt_qposadr[i]
            print(f"Joint {i}: {joint_name}, qpos_addr: {joint_adr}")

        # in __init__
        self._ctrl_target = np.zeros(self.model.nu, dtype=np.float32)

        # per-actuator step sizes (delta per env step when action=1)
        # order matches your actuators: wheels, lift, arm, wrists, gripper, head pan, head tilt
        self._delta_scale = np.array([
            0.5,  # left_wheel_vel: rad/s or similar (tune)
            0.5,  # right_wheel_vel
            0.01,  # lift position target increment og:0.01
            0.01,  # arm position target increment  og: 0.01
            0.03,  # wrist_yaw
            0.03,  # wrist_pitch
            0.03,  # wrist_roll
            0.05, # gripper  og: 0.005
            0.05,  # head_pan
            0.05,  # head_tilt
        ], dtype=np.float32)[: self.model.nu]

        self.reset()
        #---------------------------------------------------------------------------------------------------------------

    def _get_obs_internal(self):
        """Get observation without goal concatenation (for internal use)."""
        qpos = self.data.qpos

        # Robot state (5 dims): base_x, lift, arm_ext, gripper, is_grasped
        # Removed wrist_yaw, wrist_pitch, wrist_roll since they're always 0
        base_x      = qpos[IDX_BASE_X]
        lift        = qpos[IDX_LIFT]
        arm_ext     = qpos[IDX_ARM_L0] + qpos[IDX_ARM_L1] + qpos[IDX_ARM_L2] + qpos[IDX_ARM_L3]
        gripper     = qpos[IDX_GRIPPER]
        is_grasped  = self._get_grasp_flag()

        robot_state = np.array([
            base_x,
            lift, 
            arm_ext,
            gripper, 
            is_grasped
        ])

        # Object states (MAX_OBJECTS * 7 dims)
        # Only fill in num_objects, rest are zeros
        objects_state = np.zeros(self.MAX_OBJECTS * self.OBJ_FEATS, dtype=np.float64)
        
        for slot_idx, obj_id in enumerate(self.objects):
            obj_name = f"object{obj_id}"
            try:
                obj_pos = self.get_body_com(obj_name)  # (3,) get the object position
                
                # Get object quaternion from qpos
                # Assuming objects are added sequentially in XML
                obj_qpos_start = IDX_OBJ_X + obj_id * 7  # Each object freejoint has 7 qpos
                obj_qw = qpos[obj_qpos_start + 3]
                obj_qx = qpos[obj_qpos_start + 4]
                obj_qy = qpos[obj_qpos_start + 5]
                obj_qz = qpos[obj_qpos_start + 6]
                
                # Fill in object i's features: [x, y, z, qw, qx, qy, qz] (position-first)
                obj_idx = slot_idx * self.OBJ_FEATS
                objects_state[obj_idx:obj_idx + 7] = [
                    obj_pos[0], obj_pos[1], obj_pos[2],  # position first
                    obj_qw, obj_qx, obj_qy, obj_qz        # then quaternion
                ]
            except:
                # If object doesn't exist, leave as zeros
                pass

        return np.concatenate([robot_state, objects_state])
    
    def _get_obs(self):
        """Get full observation with goal (SGCRL pattern)."""
        obs = self._get_obs_internal()
        # Return [obs, goal] concatenated
        # Goal is fixed for entire training run (set on first reset)
        return np.concatenate([obs, self._goal]).astype(np.float32)

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Reset robot base state (only x position)
        qpos[IDX_BASE_X]   = 0.0
        
        # Reset robot joint positions
        qpos[IDX_LIFT]   = 0.75  # Lift body at 0.2+0.75=0.95m, allows reaching table at 0.72m
        qpos[IDX_ARM_L0] = 0.0
        qpos[IDX_ARM_L1] = 0.0
        qpos[IDX_ARM_L2] = 0.0
        qpos[IDX_ARM_L3] = 0.0
        # Fix gripper orientation (wrist joints always at 0)
        qpos[IDX_WRIST_YAW]   = 0.0
        qpos[IDX_WRIST_PITCH] = 0.0
        qpos[IDX_WRIST_ROLL]  = 0.0
        qpos[IDX_GRIPPER]     = 0.0

        self._ctrl = self.data.ctrl.copy()  # shape (nu,)

        # Reset object(s) position

        TABLE_X_CENTER = 0.0
        TABLE_Y_CENTER = -0.7
        TABLE_X_HALF   = 0.5   # < 0.64, leave margin
        TABLE_Y_HALF   = 0.25    # < 0.36, leave margin

        for obj_id in self.objects:
            obj_xy = self.np_random.uniform(
                low=np.array([TABLE_X_CENTER - TABLE_X_HALF, TABLE_Y_CENTER - TABLE_Y_HALF]),
                high=np.array([TABLE_X_CENTER + TABLE_X_HALF, TABLE_Y_CENTER + TABLE_Y_HALF]),
            )
            
            obj_qpos_start = IDX_OBJ_X + obj_id * 7
            qpos[obj_qpos_start]     = obj_xy[0]  # x
            qpos[obj_qpos_start + 1] = obj_xy[1]  # y
            qpos[obj_qpos_start + 2] = 0.75       # z
            qpos[obj_qpos_start + 3] = 1.0        # qw
            qpos[obj_qpos_start + 4] = 0.0        # qx
            qpos[obj_qpos_start + 5] = 0.0        # qy
            qpos[obj_qpos_start + 6] = 0.0        # qz

        qvel[:] = 0.0
        self.set_state(qpos, qvel) # set it into mujoco state

        mujoco.mj_forward(self.model, self.data)               # update the sim state
        self.ee_quat_ref = self._site_quat("ee_site").copy()   # Gets the end-effector orientation at the reset state
        
        # ---- Generate goal EVERY episode ----
        current_obs = self._get_obs_internal() # get current observation (without goal)

        print(f"[GOAL GENERATION DEBUG] _fixed_start_end = {self._fixed_start_end}")

        if self._fixed_start_end is not None:
            # Fixed goal (for debugging / ablation)
            print("[GOAL] Using fixed goal from _fixed_start_end parameter")
            self._goal = np.asarray(self._fixed_start_end[1], dtype=np.float32)

        else:
            print("[GOAL] Computing goal with IK solver (per-episode)")

            slot = self._slot_of_obj(self.target_obj_id)
            base = self.ROBOT_FEATS + slot * self.OBJ_FEATS
            obj_x = current_obs[base + 0]
            obj_y = current_obs[base + 1]
            obj_z = current_obs[base + 2]

            obj_z_goal = 0.9  # table 0.75 + 0.15

            print(f"[GOAL] Original object position: ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f})")
            print(f"[GOAL] Target goal position:   ({obj_x:.3f}, {obj_y:.3f}, {obj_z_goal:.3f})")

            self._goal = self._compute_goal_with_ik(self.target_obj_id, obj_x, obj_y, obj_z_goal)    # set up goal observation
            # self._goal = self._compute_goal_with_ik(obj_x, obj_y, obj_z_goal)

        # self._lift_streak = 0  # reset lift streak counter (testing2)
        
        # Initialize control targets to valid values within actuator ranges
        # For position actuators (lift, arm, wrists, gripper, head), use current qpos
        # For velocity actuators (wheels), use 0
        self._ctrl_target[0] = 0.0  # left_wheel_vel
        self._ctrl_target[1] = 0.0  # right_wheel_vel
        self._ctrl_target[2] = qpos[IDX_LIFT]      # lift position
        self._ctrl_target[3] = qpos[IDX_ARM_L0] + qpos[IDX_ARM_L1] + qpos[IDX_ARM_L2] + qpos[IDX_ARM_L3]  # arm extension
        self._ctrl_target[4] = qpos[IDX_WRIST_YAW]     # wrist_yaw
        self._ctrl_target[5] = qpos[IDX_WRIST_PITCH]   # wrist_pitch
        self._ctrl_target[6] = qpos[IDX_WRIST_ROLL]    # wrist_roll
        self._ctrl_target[7] = qpos[IDX_GRIPPER]       # gripper
        self._ctrl_target[8] = qpos[IDX_HEAD_PAN]      # head_pan
        self._ctrl_target[9] = qpos[IDX_HEAD_TILT]     # head_tilt
        
        # Apply the control targets
        self.data.ctrl[:] = self._ctrl_target.copy()


        return self._get_obs()

    def _slot_of_obj(self, obj_id: int) -> int:  # decide which slot an object is in
        return self.objects.index(obj_id)  

    def step(self, a):
        # print("[ENV] received:", np.round(np.asarray(a)[:], 3))  # DEBUG
        a_raw = np.asarray(a).copy()
        
        # Fix gripper orientation: zero out wrist yaw, pitch, roll actions
        # Action indices match actuator order (now 11 actuators): 
        # 0: left_wheel, 1: right_wheel, 2: lift, 3: arm, 
        # 4: wrist_yaw, 5: wrist_pitch, 6: wrist_roll, 7: gripper, 8: head_pan, 9: head_tilt
        a = np.copy(a)  # Don't modify the original action
        
        a = np.clip(a, -1.0, 1.0)  # Ensure action is within [-1, 1]
        # print("raw:", np.round(a_raw[:], 3), "clipped:", np.round(a[:], 3))  # DEBUG
        a[4:7] = 0.0  # zero out wrist actions
        # a[7] = 0.05   # slightly open gripper every step
        a[8:10] = 0.0  # no head movement
        # a[2:4] = 0.0  # no lift/arm movement for now (testing)
        # v = 0.5 * (a[0] + a[1])  # average wheel velocity
        # a[0] = v
        # a[1] = v
        
        # wheels: set directly (do not integrate)
        v = float(a[0])
        v = float(np.clip(v, -1.0, 1.0))

        vmax = min(self.model.actuator_ctrlrange[0,1], self.model.actuator_ctrlrange[1,1])

        self._ctrl_target[0] = v * vmax
        self._ctrl_target[1] = v * vmax

        # then zero those entries so they won't be integrated below
        a[0] = 0.0
        a[1] = 0.0
        # print("[ENV] final used:", np.round(a[:], 3))  # DEBUG
        self._ctrl_target += a * self._delta_scale  # scale action by per-actuator delta (_ctrl_target is the command before filtering)
        # print("[WHEELS] v=", v, "ctrl_target=", self._ctrl_target[0], self._ctrl_target[1], "data.ctrl=", self.data.ctrl[0], self.data.ctrl[1])  # DEBUG

        low, high = self.model.actuator_ctrlrange.T
        self._ctrl_target = np.clip(self._ctrl_target, low, high)

        # Apply first-order low-pass filter to control signals
        alpha = 0.2  # 0.05 smoother, 0.3 more responsive
        self.data.ctrl[:] = (1 - alpha) * self.data.ctrl[:] + alpha * self._ctrl_target # data.ctrl is the command after filtering

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

       # self.do_simulation(np.clip(self._ctrl, low, high), self.frame_skip)  # take action
        
       
        # self.data.qpos[IDX_WRIST_YAW] = 0.0
        # self.data.qpos[IDX_WRIST_PITCH] = 0.0
        # self.data.qpos[IDX_WRIST_ROLL] = 0.0
        # self.data.qpos[IDX_GRIPPER] = 0.05
        # mujoco.mj_forward(self.model, self.data)
        
        if self.render_mode == "human":
            self.render()
            # This prints every step when render_mode=="human"
            # print(f"[RENDER] Step {self.data.time:.2f}s - Rendering frame")  # Debug print

        full_obs = self._get_obs()
        
        # Extract current observation (first half of full_obs)
        obs_dim = self.ROBOT_FEATS + self.MAX_OBJECTS * self.OBJ_FEATS
        current_obs = full_obs[:obs_dim]  # ONLY current state
        # goal_obs    = full_obs[obs_dim:] # for (testing1)
        
        # Extract first object's position from current observation
        # Object state format: [x, y, z, qw, qx, qy, qz] (position-first)
        slot = self._slot_of_obj(self.target_obj_id)
        base = self.ROBOT_FEATS + slot * self.OBJ_FEATS
        # obj_x = current_obs[base + 0]  # Not used for success check
        # obj_y = current_obs[base + 1]  # Not used for success check
        # obj_z = current_obs[base + 2]  # Not used for success check
        is_grasped = current_obs[self.ROBOT_FEATS - 1]  # is_grasped is last robot feature
        # base_x = current_obs[0]      # robot base_x (testing1)
        # goal_base_x = goal_obs[0]    # goal base_x  (testing1)
        
        # obj_pos = np.array([obj_x, obj_y, obj_z])  # Removed: using fresh obj_pos_wc instead

        obj_pos_wc = self.get_body_com(f"object{self.target_obj_id}").copy()  # (testing3) get updated object position from sim

        ee_site_id = self.model.site("ee_site").id    # Get End-Effector position (testing3)
        ee_pos = self.data.site_xpos[ee_site_id].copy()  # (testing3) Get updated EE position from sim
        dx = ee_pos[0] - obj_pos_wc[0]   # x alignment。(testing3)
        dy = ee_pos[1] - obj_pos_wc[1]   # y relative (front/back) (testing3)
        
        # # ---------------------------------------(testing3)-------------------------------------
        X_TOL = 0.03          # 3cm 
        FRONT_MARGIN = 0.05   # 5cm

        x_aligned = abs(dx) <= X_TOL 
        in_front = (dy >= -FRONT_MARGIN) and (dy < 0.0)
        # Threshold for alignment (e.g., 0.05m = 5cm)
        success = bool(x_aligned and in_front)
        r = float(success)

        info = {
            "success": success,
            "dx": float(dx),
            "dy": float(dy),
            "x_aligned": bool(x_aligned),
            "in_front": bool(in_front),
        }
        terminated = False
        truncated = False
        # ---------------------------------------(testing3)-------------------------------------
        #---------------------------------------(testing1)-------------------------------------
        # X_TOL = 0.05                 # (testing1)
        # success = bool(abs(base_x - goal_base_x) < X_TOL) # (testing1)
        # r = float(success) # (testing1)
        # info = {"success": success, "base_x": float(base_x), "goal_base_x": float(goal_base_x)} # (testing1)
        # terminated = False # (testing1)
        # truncated = False  # (testing1)

        # --------update lift streak (is_grasped is 0/1) (testing2)-----------------------------------
        # --- lift streak (anti-bounce) ---
        # LIFT_THRESH = 0.85
        # if obj_pos[2] > LIFT_THRESH:
        #     self._lift_streak += 1
        # else:
        #     self._lift_streak = 0

        # stable_lift = (self._lift_streak >= self._lift_streak_required)

        # # Phase 1: only require stable lift
        # success = bool(stable_lift)

        # # Phase 2: stable lift + grasp
        # # success = bool(stable_lift and (is_grasped > 0.5))

        # r = float(success)

        # info = {
        #     "success": success,
        #     "obj_pos": obj_pos,
        #     "is_grasped": float(is_grasped),
        #     "lift_streak": int(self._lift_streak),
        #     "stable_lift": bool(stable_lift),
        # }
        # terminated = False
        # truncated = False
        
        #---------------------------------------(testing2)-------------------------------------
        #---------------------------------------(origianl)-------------------------------------
        # Use fresh object position from simulation (after mj_step)
        # obj_pos_wc = self.get_body_com(f"object{self.target_obj_id}").copy()
        
        # LIFT_THRESH = 0.85  # 10cm above table (table at 0.72m, objects start at 0.75m)
        
        # # Success = object lifted AND grasped by robot
        # success = bool(
        #     (obj_pos_wc[2] > LIFT_THRESH) and  # Object height check
        #     (is_grasped > 0.5)                   # Grasp check (both fingers touching)
        # )

        # # SGCRL pattern: episodes don't terminate on success, only on max_episode_steps
        # terminated = False
        # truncated = False

        # r = float(success)

        # info = {
        #     "success": success,
        #     "obj_pos": obj_pos_wc,  # Report fresh position
        #     "obj_height": float(obj_pos_wc[2]),
        #     "is_grasped": float(is_grasped),
        # }
        #---------------------------------------(origianl)-------------------------------------

        return full_obs, r, terminated, truncated, info
    
    def _get_grasp_flag(self) -> float:
        object_geom_id = self.object_geom_ids.get(self.target_obj_id, None)
        if object_geom_id is None or len(self.finger_geom_ids) < 2:
            return 0.0

        left_id, right_id = self.finger_geom_ids[0], self.finger_geom_ids[1]
        left_touch = False
        right_touch = False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            if (g1 == object_geom_id and g2 == left_id) or (g2 == object_geom_id and g1 == left_id):
                left_touch = True
            if (g1 == object_geom_id and g2 == right_id) or (g2 == object_geom_id and g1 == right_id):
                right_touch = True

            if left_touch and right_touch:
                return 1.0

        return 0.0

    def _safe_render(self, n=1):
    # In launchpad/acme actor workers, self.spec is often None -> render() asserts.
        if getattr(self, "spec", None) is None:
            return
        for _ in range(n):
            try:
                self.render()
            except AssertionError:
                return
        
    def _site_quat(self, site_name: str) -> np.ndarray:
            sid = self.model.site(site_name).id
            mat = self.data.site_xmat[sid].copy()          # (9,) 3*3
            quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat, mat)                 # mat is 9 floats
            return quat


    def _compute_goal_with_ik(self, obj_id, obj_x, obj_y, obj_z_goal):
        """Compute goal observation using IK solver to place end-effector at target object position."""
        # 1. Save state
        qpos0 = self.data.qpos.copy()
        qvel0 = self.data.qvel.copy()
        
        # 2. Set mocap target
        mocap_id = int(self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target0")
        ])

        self.data.mocap_pos[mocap_id] = np.array([obj_x, obj_y, obj_z_goal]) # move the mocap body to the target position
        self.data.mocap_quat[mocap_id] = self.ee_quat_ref.copy() # set the mocap orientation to the reference end-effector orientation
        mujoco.mj_forward(self.model, self.data)
        
        # 3. Update IK and solve
        self.ik.update_configuration(self.data.qpos)
        success = self.ik.converge_ik(dt=0.01)  # move te ee to the target position

        # Apply solution
        self.data.qpos[:] = self.ik.configuration.q
        mujoco.mj_forward(self.model, self.data)

        # Check actual error
        ee_site_id = self.model.site("ee_site").id
        ee_pos = self.data.site_xpos[ee_site_id].copy()
        target_pos = self.data.mocap_pos[mocap_id].copy()
        ik_error = np.linalg.norm(target_pos - ee_pos)

        print(f"[IK] MinkIK success={success}, actual EE error={ik_error:.6f}m")

        # 4. Check BOTH conditions: IK convergence AND actual error
        IK_ERROR_THRESHOLD = 0.01  # 1cm tolerance
        
        if not success or ik_error > IK_ERROR_THRESHOLD:
            # IK failed OR error too large
            if not success:
                print(f"[IK] REJECTED - IK did not converge")
            if ik_error > IK_ERROR_THRESHOLD:
                print(f"[IK] REJECTED - error {ik_error:.4f}m > {IK_ERROR_THRESHOLD}m")
            
            # 5. Resample if failed - sample from ENTIRE table
            self.set_state(qpos0, qvel0)
            # self.data.mocap_pos[:] = mocap_pos0
            mujoco.mj_forward(self.model, self.data)
            
            # Resample from entire table workspace
            TABLE_X_CENTER = 0.0
            TABLE_Y_CENTER = -0.7
            TABLE_X_HALF   = 0.5
            TABLE_Y_HALF   = 0.25
            
            obj_xy_new = self.np_random.uniform(
                low=np.array([TABLE_X_CENTER - TABLE_X_HALF, TABLE_Y_CENTER - TABLE_Y_HALF]),
                high=np.array([TABLE_X_CENTER + TABLE_X_HALF, TABLE_Y_CENTER + TABLE_Y_HALF]),
            )
            obj_x_new = obj_xy_new[0]
            obj_y_new = obj_xy_new[1]
            
            if not hasattr(self, '_ik_attempts'):
                self._ik_attempts = 0
            self._ik_attempts += 1
            
            if self._ik_attempts > 5:
                print("[IK] Max attempts reached, using fallback goal")
                self._ik_attempts = 0
                return self._create_fallback_goal(obj_id, obj_x, obj_y, obj_z_goal)
            
            print(f"[IK] Retry {self._ik_attempts}/5 with new position ({obj_x_new:.3f}, {obj_y_new:.3f})")
            result = self._compute_goal_with_ik(obj_id, obj_x_new, obj_y_new, obj_z_goal)
            self._ik_attempts = 0
            return result

        # If we reach here: IK succeeded AND error is acceptable
        print(f"[IK] ACCEPTED - success={success}, error={ik_error:.6f}m")
        self._ik_attempts = 0  # Reset on success

        self._last_successful_ik_qpos = self.data.qpos.copy()  # Cache successful IK solution
        print("[IK] Cached successful IK solution for future fallbacks")

        # Continue with goal generation (no need to re-apply IK solution)
        
        # 6. Move object to goal position
        obj_qpos_start = IDX_OBJ_X + obj_id * 7
        self.data.qpos[obj_qpos_start + 0] = obj_x
        self.data.qpos[obj_qpos_start + 1] = obj_y
        self.data.qpos[obj_qpos_start + 2] = obj_z_goal

        # 7. Enforce wrist orientation
        self.data.qpos[IDX_WRIST_YAW] = 0.0
        self.data.qpos[IDX_WRIST_PITCH] = 0.0
        self.data.qpos[IDX_WRIST_ROLL] = 0.0
        # mujoco.mj_forward(self.model, self.data)
        
        # Close gripper
        # self.data.qpos[IDX_GRIPPER] = -0.1
        mujoco.mj_forward(self.model, self.data)

        # 8. Get observation
        goal_obs = self._get_obs_internal()
        goal_obs[self.ROBOT_FEATS - 1] = 1.0  # Force is_grasped flag
        
        # Debug output
        ee_pos_final = self.data.site_xpos[ee_site_id].copy()
        print("=== IK DEBUG ===")
        print(f"Target (mocap) pos : {target_pos}")
        print(f"Gripper (EE) pos   : {ee_pos_final}")
        print(f"EE error (L2)      : {np.linalg.norm(target_pos - ee_pos_final):.6f}m")
        print("================")

        print("=== IK GOAL OBS ===")
        print("shape:", goal_obs.shape)
        print(goal_obs)
        print("ee_site pos:", self.data.site_xpos[self.model.site("ee_site").id])
        print("===================")

        # 9. Restore state
        self.set_state(qpos0, qvel0)
        mujoco.mj_forward(self.model, self.data)

        return goal_obs

    def _create_fallback_goal(self, obj_id, obj_x, obj_y, obj_z_goal):
        """Create a physically plausible fallback goal using cached IK solution."""
        
        if self._last_successful_ik_qpos is not None:
            print("[IK] Using last successful IK solution as fallback")
            fallback_qpos = self._last_successful_ik_qpos.copy()
        elif self._default_reach_qpos is not None:
            print("[IK] Using default reach pose as fallback")
            fallback_qpos = self._default_reach_qpos.copy()
        else:
            print("[IK] No cached solution, using predefined reach pose")
            fallback_qpos = self._get_predefined_reach_pose()
        
        # Store current state
        qpos0 = self.data.qpos.copy()
        qvel0 = self.data.qvel.copy()
        
        # Apply fallback robot configuration
        self.data.qpos[:] = fallback_qpos
        
        # Move object to goal position
        obj_qpos_start = IDX_OBJ_X + obj_id * 7
        self.data.qpos[obj_qpos_start + 0] = obj_x
        self.data.qpos[obj_qpos_start + 1] = obj_y
        self.data.qpos[obj_qpos_start + 2] = obj_z_goal
        
        # Enforce wrist orientation
        self.data.qpos[IDX_WRIST_YAW] = 0.0
        self.data.qpos[IDX_WRIST_PITCH] = 0.0
        self.data.qpos[IDX_WRIST_ROLL] = 0.0
        
        # mujoco.mj_forward(self.model, self.data)
        
        # Close gripper
        # self.data.qpos[IDX_GRIPPER] = -0.1
        mujoco.mj_forward(self.model, self.data)
        
        # Get observation
        goal_obs = self._get_obs_internal()
        goal_obs[self.ROBOT_FEATS - 1] = 1.0
        
        # Restore original state
        self.set_state(qpos0, qvel0)
        mujoco.mj_forward(self.model, self.data)
        
        print("[IK] Fallback goal created successfully")
        return goal_obs
    
    def _get_predefined_reach_pose(self):
        """Return a reasonable reaching configuration as last resort."""
        qpos = self.init_qpos.copy()
        
        # Predefined reaching configuration (adjust based on your robot)
        qpos[IDX_BASE_X] = 0.0
        qpos[IDX_LIFT] = 0.78        # Lift up
        qpos[IDX_ARM_L0] = 0.12      # Extend arm partially
        qpos[IDX_ARM_L1] = 0.12
        qpos[IDX_ARM_L2] = 0.12
        qpos[IDX_ARM_L3] = 0.12
        qpos[IDX_WRIST_YAW] = 0.0
        qpos[IDX_WRIST_PITCH] = 0.0
        qpos[IDX_WRIST_ROLL] = 0.0
        qpos[IDX_GRIPPER] = 0.0     # Open gripper
        
        print("[IK] Using hardcoded predefined reach pose")
        return qpos


