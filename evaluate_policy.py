import time
import numpy as np
import gymnasium as gym

def run_eval(env, policy_fn, episodes=20, max_steps=200, render=False):
    successes = []
    returns = []

    for ep in range(episodes):
        obs, info = env.reset()
        ep_ret = 0.0
        ep_success = False

        for t in range(max_steps):
            action = policy_fn(obs)   # deterministic action
            obs, r, terminated, truncated, info = env.step(action)
            ep_ret += float(r)
            ep_success = ep_success or bool(info.get("success", False))

            if render:
                env.render()

            if terminated or truncated:
                break

        returns.append(ep_ret)
        successes.append(1.0 if ep_success else 0.0)
        print(f"[eval] ep={ep:03d} return={ep_ret:.2f} success={ep_success}")

    print("==== Evaluation Summary ====")
    print(f"episodes: {episodes}")
    print(f"success_rate: {np.mean(successes):.3f}")
    print(f"avg_return:   {np.mean(returns):.3f}")
    print("============================")

def main():
    # 1) build env (use human or rgb_array)
    # If headless/video: render_mode="rgb_array"
    from envs.stretch_pick_env import StretchPickEnv  # adjust import
    env = StretchPickEnv(render_mode="human")

    # 2) load policy from checkpoint
    learner_dir = "logs/contrastive_cpc_stretch_pick_42/3af967c4-dfe1-11f0-aa29-b44506f1d30/checkpoints/learner"

    # TODO: implement based on your codebase:
    policy_fn = load_policy_fn(learner_dir)

    run_eval(env, policy_fn, episodes=20, max_steps=200, render=True)
    env.close()

if __name__ == "__main__":
    main()
