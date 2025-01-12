# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import os
from typing import Union, Dict, Any, Optional
import numpy as np
from xpag.agents.agent import Agent
from xpag.setters.setter import Setter
from xpag.tools.utils import DataType, datatype_convert, hstack, logical_or
from xpag.tools.timing import timing
from xpag.tools.logging import eval_log
from xpag.plotting.plotting import single_episode_plot
from xpag.plotting.plotting import plot_achieved_goals, update_csv


class SaveEpisode:
    """To save episodes in Brax or Mujoco environments"""

    def __init__(self, env, env_info):
        self.env = env
        self.env_info = env_info
        self.qpos = []
        self.qrot = []
        self.qvel = []
        self.qang = []
        self.target_pos = []


    def update(self):
        if self.env_info["env_type"] == "Brax":
            self.qpos.append(self.env.unwrapped._state.qp.pos.to_py())
            self.qrot.append(self.env.unwrapped._state.qp.rot.to_py())
            self.qvel.append(self.env.unwrapped._state.qp.vel.to_py())
            self.qang.append(self.env.unwrapped._state.qp.ang.to_py())
        elif self.env_info["env_type"] == "Mujoco":
            posvel = np.split(
                np.array(self.env.call("state_vector")),
                [self.env.call("init_qpos")[0].shape[-1]],
                axis=1,
            )
            self.qpos.append(posvel[0])
            self.qvel.append(posvel[1])

        elif self.env_info["env_type"] == "Fetch":
            qpos = np.copy(self.env.data.qpos)
            qvel = np.copy(self.env.data.qvel)
            target_pos = np.copy(self.env.goal)

            self.qpos.append(qpos)
            self.qvel.append(qvel)
            self.target_pos.append(target_pos)
        else:
            pass

    def save(self, i: int, save_dir: str):
        os.makedirs(os.path.join(save_dir, "episode"), exist_ok=True)
        if self.env_info["env_type"] == "Brax":
            with open(os.path.join(save_dir, "episode", "env_name.txt"), "w") as f:
                print(self.env_info["name"], file=f)
            np.save(
                os.path.join(save_dir, "episode", "qp_pos"),
                [pos[i] for pos in self.qpos],
            )
            np.save(
                os.path.join(save_dir, "episode", "qp_rot"),
                [rot[i] for rot in self.qrot],
            )
            np.save(
                os.path.join(save_dir, "episode", "qp_vel"),
                [vel[i] for vel in self.qvel],
            )
            np.save(
                os.path.join(save_dir, "episode", "qp_ang"),
                [ang[i] for ang in self.qang],
            )
        elif self.env_info["env_type"] == "Mujoco":
            with open(os.path.join(save_dir, "episode", "env_name.txt"), "w") as f:
                print(self.env_info["name"], file=f)
            np.save(
                os.path.join(save_dir, "episode", "qpos"), [pos[i] for pos in self.qpos]
            )
            np.save(
                os.path.join(save_dir, "episode", "qvel"), [vel[i] for vel in self.qvel]
            )

        elif self.env_info["env_type"] == "Fetch":
            with open(os.path.join(save_dir, "episode", "env_name.txt"), "w") as f:
                print(self.env_info["name"], file=f)
            np.save(
                os.path.join(save_dir, "episode", "qpos"), self.qpos
            )
            np.save(
                os.path.join(save_dir, "episode", "qvel"), self.qvel
            )
            np.save(
                os.path.join(save_dir, "episode", "target"), self.target_pos
            )
            
        #import ipdb;ipdb.set_trace()

        self.qpos = []
        self.qrot = []
        self.qvel = []
        self.qang = []


def single_rollout_eval(
    steps: int,
    eval_env: Any,
    env_info: Dict[str, Any],
    agent: Agent,
    setter: Setter,
    save_dir: Union[str, None] = None,
    plot_projection=None,
    save_episode: bool = False,
    env_datatype: Optional[DataType] = None,
    seed: Optional[int] = None,
    log: bool = True,
    force_goal: np.array = None
):
    """Evaluation performed on a single run"""
    master_rng = np.random.RandomState(
        seed if seed is not None else np.random.randint(1e9)
    )
    interval_time, _ = timing()
    observation, _ = setter.reset(
        eval_env,
        *eval_env.reset(seed=master_rng.randint(1e9)),
        eval_mode=True,
    )           
    
    if force_goal is not None:
        eval_env.goal = np.copy(force_goal)
        observation['desired_goal'] = np.copy(force_goal)
    
    if save_episode and save_dir is not None:
        save_ep = SaveEpisode(eval_env, env_info)
        save_ep.update()
           
    done = np.array(False)
    cumulated_reward = 0.0
    step_list = []
    while not done.max():
        obs = (
            observation
            if not env_info["is_goalenv"]
            else hstack(observation["observation"], observation["desired_goal"])
        )
        action = agent.select_action(obs, eval_mode=True)
        
        action_info = {}
        if isinstance(action, tuple):
            action_info = action[1]
            action = action[0]
        action = datatype_convert(action, env_datatype)
        next_observation, reward, terminated, truncated, info = setter.step(
            eval_env,
            observation,
            action,
            action_info,
            *eval_env.step(action),
            eval_mode=True,
        )
        done = logical_or(terminated, truncated)
        if save_episode and save_dir is not None:
            save_ep.update()
        cumulated_reward += reward.mean()
        step_list.append(
            {"observation": observation, "next_observation": next_observation}
        )
        observation = next_observation
        if force_goal is not None:
            eval_env.goal = np.copy(force_goal)
            observation['desired_goal'] = np.copy(force_goal)
    if log:
        eval_log(
            steps,
            interval_time,
            cumulated_reward,
            None if not env_info["is_goalenv"] else info["is_success"].mean(),
            env_info,
            agent,
            save_dir,
        )
    
    obs_list = None
    if plot_projection is not None and save_dir is not None:
        os.makedirs(os.path.join(os.path.expanduser(save_dir), "plots", "episodes"), exist_ok=True)
        obs_list = single_episode_plot(
            os.path.join(
                os.path.expanduser(save_dir),
                "plots", "episodes",
                f"{steps:12}.png".replace(" ", "0"),
            ),
            step_list,
            projection_function=plot_projection,
            plot_env_function=None if not hasattr(eval_env, "plot") else eval_env.plot,
        )
    if save_episode and save_dir is not None:
        save_ep.save(0, os.path.expanduser(save_dir))
    timing()
    
    return obs_list





def multiple_rollout_eval(
    steps: int,
    eval_env: Any,
    env_info: Dict[str, Any],
    agent: Agent,
    setter: Setter,
    save_dir: Union[str, None] = None,
    env_datatype: Optional[DataType] = None,
    seed: Optional[int] = None,
    goal_fn: np.array = None,
    writer = None
):
    """Evaluation performed on a single run"""
    master_rng = np.random.RandomState(
        seed if seed is not None else np.random.randint(1e9)
    )
    observation, _ = setter.reset(
        eval_env,
        *eval_env.reset(seed=master_rng.randint(1e9)),
        eval_mode=True,
    )
    
    if goal_fn is not None:
        # WARNING ! Only valid for sibrivalry custom maze env
        num_goals =  observation["desired_goal"].shape[0]
        env_size = env_info["maze_size"]
        num_grid = int((env_size[0]+1)**2)
        num_goals_per_grid = num_goals//num_grid
        desired_goals = goal_fn(num_goals_per_grid)
        
        if env_info["from_pixels"]:
            init_goals = desired_goals.copy()
            desired_goals = eval_env.convert_2D_to_embed(desired_goals)
        
        assert observation["desired_goal"].shape == desired_goals.shape
    else:
        desired_goals = observation["desired_goal"]
        
    done = np.array([False]*eval_env.num_envs) 
    while not np.all(done):
        obs = (
            observation
            if not env_info["is_goalenv"]
            else hstack(observation["observation"], desired_goals)
        )
        action = agent.select_action(obs, eval_mode=True)
        action_info = {}
        if isinstance(action, tuple):
            action_info = action[1]
            action = action[0]
        action = datatype_convert(action, env_datatype)
        next_observation, reward, terminated, truncated, info = setter.step(
            eval_env,
            observation,
            action,
            action_info,
            *eval_env.step(action),
            eval_mode=True,
        )
        
        if goal_fn is not None:
            if env_info["from_pixels"]:
                dist = np.linalg.norm(next_observation["init_obs"] - init_goals, axis=-1)
            else:    
                dist = np.linalg.norm(next_observation["achieved_goal"] - desired_goals, axis=-1)
            succ = (dist < 0.15).reshape(-1,1)
            terminated = np.copy(succ)
            info["is_success"] = np.copy(succ)
            
              
        done = logical_or(terminated, truncated)
        observation = next_observation
    
    
    update_csv("average success", info["is_success"].mean(), steps, save_dir, writer)
    return info["is_success"].astype('float32')