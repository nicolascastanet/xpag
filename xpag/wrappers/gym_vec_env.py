# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import sys
import inspect
from typing import Callable
import numpy as np
import torch
import gym
from gym.vector.utils import write_to_shared_memory, concatenate, create_empty_array
from gym.vector import VectorEnv, AsyncVectorEnv
from xpag.wrappers.reset_done import ResetDoneWrapper
from xpag.tools.utils import get_env_dimensions
import copy



def check_goalenv(env) -> bool:
    """
    Checks if an environment is of type 'GoalEnv'.
    The migration of GoalEnv from gym (0.22) to gym-robotics makes this verification
    non-trivial. Here we just verify that the observation_space has a structure
    that is compatible with the GoalEnv class.
    """
    if isinstance(env, VectorEnv):
        obs_space = env.single_observation_space
    else:
        obs_space = env.observation_space
    if not isinstance(obs_space, gym.spaces.Dict):
        return False
    else:
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key not in obs_space.spaces:
                return False
    return True


def gym_vec_env_(env_name, num_envs, wrap_function=None):
    if wrap_function is None:

        def wrap_function(x):
            return x

    if "num_envs" in inspect.signature(
        gym.envs.registration.load(gym.spec(env_name).entry_point).__init__
    ).parameters and hasattr(
        gym.envs.registration.load(gym.spec(env_name).entry_point),
        "reset_done",
    ):
        # no need to create a VecEnv and wrap it if the env accepts 'num_envs' as an
        # argument at __init__ and has a reset_done() method. In this case, we trust
        # the environment to properly handle parallel rollouts.

        env = wrap_function(
            gym.make(env_name, num_envs=num_envs).unwrapped  # removing gym wrappers
        )

        # We force the environment to have a time limit, but
        # env.spec.max_episode_steps cannot exist as it would automatically trigger
        # the TimeLimit wrapper of gym, which does not handle batch envs. We require
        # max_episode_steps to be stored as an attribute of env:
        assert (
            (
                not hasattr(env.spec, "max_episode_steps")
                or env.spec.max_episode_steps is None
            )
            and hasattr(env, "max_episode_steps")
            and env.max_episode_steps is not None
        ), (
            "Trying to create a batch environment. env.max_episode_steps must exist, "
            "and env.spec.max_episode_steps must not (or be None)."
        )
        max_episode_steps = env.max_episode_steps
        env_type = "Gym"
    else:
        dummy_env = gym.make(env_name)
        # We force the env to have either a standard gym time limit (with the max number
        # of steps defined in .spec.max_episode_steps), or the max number of steps
        # defined in .max_episode_steps (and in this case we trust the environment
        # to appropriately prevent episodes from exceeding max_episode_steps steps).
        assert (
            hasattr(dummy_env.spec, "max_episode_steps")
            and dummy_env.spec.max_episode_steps is not None
        ) or (
            hasattr(dummy_env, "max_episode_steps")
            and dummy_env.max_episode_steps is not None
        ), (
            "Only allowing gym envs with time limit (defined in "
            ".spec.max_episode_steps or .max_episode_steps)."
        )
        if (
            hasattr(dummy_env.spec, "max_episode_steps")
            and dummy_env.spec.max_episode_steps is not None
        ):
            max_episode_steps = dummy_env.spec.max_episode_steps
        else:
            max_episode_steps = dummy_env.max_episode_steps
        # env_type = "Mujoco" if isinstance(dummy_env.unwrapped, MujocoEnv) else "Gym"
        # To avoid imposing a dependency to mujoco, we simply guess that the
        # environment is a mujoco environment when it has the 'init_qpos', 'init_qvel',
        # 'state_vector', 'do_simulation' and 'get_body_com' attributes:
        env_type = (
            "Mujoco"
            if hasattr(dummy_env.unwrapped, "init_qpos")
            and hasattr(dummy_env.unwrapped, "init_qvel")
            and hasattr(dummy_env.unwrapped, "state_vector")
            and hasattr(dummy_env.unwrapped, "do_simulation")
            and hasattr(dummy_env.unwrapped, "get_body_com")
            else "Gym"
        )
        # The 'init_qpos' and 'state_vector' attributes are the one required to
        # save mujoco episodes (cf. class SaveEpisode in xpag/tools/eval.py).
        env = wrap_function(
            ResetDoneVecWrapper(
                AsyncVectorEnv(
                    [
                        (lambda: gym.make(env_name))
                        if hasattr(dummy_env, "reset_done")
                        else (lambda: ResetDoneWrapper(gym.make(env_name)))
                    ]
                    * num_envs,
                    worker=_worker_shared_memory_no_auto_reset,
                ),
                max_episode_steps,
            )
        )

    is_goalenv = check_goalenv(env)
    env_info = {
        "env_type": env_type,
        "name": env_name,
        "is_goalenv": is_goalenv,
        "num_envs": num_envs,
        "max_episode_steps": env.max_episode_steps,
        "action_space": env.action_space,
        "single_action_space": env.single_action_space,
    }
    get_env_dimensions(env_info, is_goalenv, env)
    return env, env_info


def gym_vec_env(env_name: str, num_envs: int, wrap_function: Callable = None):
    env, env_info = gym_vec_env_(env_name, num_envs, wrap_function)
    eval_env, _ = gym_vec_env_(env_name, 1, wrap_function)
    return env, eval_env, env_info



#def make_async_env(env_fn, dummy_env, num_envs, max_ep_steps):
#    env = ResetDoneVecWrapper(
#                AsyncVectorEnv(
#                    [env_fn]
#                    * num_envs,
#                    worker=_worker_shared_memory_no_auto_reset,
#                ),
#                max_ep_steps,
#            )
#    return env


def make_async_env(env_fn, num_envs):
    env = AsyncVectorEnv(
                    [env_fn]
                    * num_envs,
                    worker=_worker_shared_memory_no_auto_reset,
                )
    return env


def custom_vec_env(
                env_fn, 
                eval_env_fn, 
                max_ep_steps: int, 
                num_envs_train: int, 
                num_envs_eval_mult: int, 
                name:str="2D_maze",
                embed=None
            ):
    
    eval_env = eval_env_fn()
    dummy_env = copy.deepcopy(eval_env)
    
    env = make_async_env(env_fn, num_envs_train)
    
    is_goalenv = check_goalenv(env)
    
    eval_env_mult = make_async_env(eval_env_fn, num_envs_eval_mult)
    
    if embed is not None:
        env = EmbedVecWrapper(env, dummy_env, embed)
        eval_env = EmbedVecWrapper(dummy_env, dummy_env, embed)
        eval_env_mult = EmbedVecWrapper(eval_env_mult, dummy_env, embed)
        
    env = ResetDoneVecWrapper(env,max_ep_steps)
    eval_env = ResetDoneVecWrapper(eval_env,max_ep_steps)
    eval_env_mult = ResetDoneVecWrapper(eval_env_mult,max_ep_steps)
    
        
    env_info = {
        "env_type": "custom",
        "name": name,
        "is_goalenv": is_goalenv,
        "num_envs": num_envs_train,
        "max_episode_steps": env.max_episode_steps,
        "action_space": env.action_space,
        "single_action_space": eval_env.action_space,
        "maze_size":eval_env.size_max,
        "from_pixels": bool(embed)
    }
    
    get_env_dimensions(env_info, is_goalenv, env)

    return env, eval_env, eval_env_mult, env_info


class EmbedVecWrapper(gym.ObservationWrapper):
    """_summary_
    Pixel obs Wrapper for 2D Sibrivalry maze
    Args:
        gym (_type_): _description_
    """
    def __init__(self, env: VectorEnv, dummy_env, embed):
        super().__init__(env)
        self.env = env
        self.embed = embed
        self.latent_dim = embed.latent_dim 
        self.rgb_maze = dummy_env.get_rgb_maze()
        
        if not hasattr(self, "num_envs"):
            self.num_envs = 1
        
    def observation(self, obs):
        observation = obs['observation']
        desired_goal = obs['desired_goal']
        achieved_goal = obs['achieved_goal']
        
        new_keys = {'observation':'init_obs', 'desired_goal':'init_dg', 'achieved_goal':'init_ag'}
        obs = dict((new_keys[key], value) for (key, value) in obs.items())
        
        batch_obs = np.concatenate((observation, desired_goal, achieved_goal)).reshape(-1,2)
        batch_pixels_obs = self.torch(np.swapaxes(self.convert_2D_to_pixel(batch_obs),3,1))
        
        with torch.no_grad():
            batch_latent_obs = self.embed(batch_pixels_obs)
        
        #pixel_obs, pixel_dg, _ = torch.split(batch_pixels_obs, self.num_envs)
        observation, desired_goal, achieved_goal = torch.split(batch_latent_obs, self.num_envs)
        
        if self.num_envs == 1:
            observation = observation.squeeze()
            desired_goal = desired_goal.squeeze()
            achieved_goal = achieved_goal.squeeze()
            
        else:
            assert observation.shape == torch.Size([self.num_envs, self.latent_dim])
            assert desired_goal.shape == torch.Size([self.num_envs, self.latent_dim])
            assert achieved_goal.shape == torch.Size([self.num_envs, self.latent_dim])
                
        # reconvert to numpy
        # Latent space is used for classic RL operations
        obs['observation'] = observation.numpy()
        obs['desired_goal'] = desired_goal.numpy()
        obs['achieved_goal'] = achieved_goal.numpy()
        
        return obs

        
    def convert_2D_to_pixel(self, obs, rg=7):
        
        x_obs = (65-obs[:,-1]*11.4).astype(int)
        y_obs = (20+obs[:,0]*11.4).astype(int)
        coord = np.column_stack((y_obs, x_obs))
        fig_array_pixels = np.tile(self.rgb_maze[np.newaxis, ...], 
                                      (obs.shape[0], 1, 1, 1)
                                    )
            
        # Calculate the boundaries for the surrounding pixels
        left = np.maximum(coord[:, 0] - rg, 0)
        right = np.minimum(coord[:, 0] + rg, fig_array_pixels.shape[2])
        top = np.maximum(coord[:, 1] - rg, 0)
        bottom = np.minimum(coord[:, 1] + rg, fig_array_pixels.shape[1])
        
        mask = np.ones((rg*2, rg*2), dtype=bool)
        
        for i in range(len(coord)):
            fig_array_pixels[i, top[i]:bottom[i], left[i]:right[i]][mask] = [255, 0, 0]  # Example modification
            
        return fig_array_pixels / 256
    
    
    def convert_2D_to_embed(self, obs):
        """
        input obs: np.array
        """
        batch_pixels_obs = self.torch(np.swapaxes(self.convert_2D_to_pixel(obs.reshape(-1,2)),3,1))
        with torch.no_grad():
            batch_latent_obs = self.embed(batch_pixels_obs)
            
        if obs.shape[0] == 1:
            batch_latent_obs = batch_latent_obs.squeeze()
                
        return batch_latent_obs.numpy()
                
                
    def torch(self, x):
        x = torch.from_numpy(x)
        x = x.type(torch.float)
        return x
  
    


class ResetDoneVecWrapper(gym.Wrapper):
    def __init__(self, env: VectorEnv, max_episode_steps: int):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.num_envs = (self.env.num_envs if hasattr(self.env, "num_envs")
                        else 1)

    def reset(self, **kwargs):
        obs, info_ = self.env.reset(**kwargs)
        return obs, {"info_tuple": tuple(info_)}

    def reset_done(self, *args, **kwargs):
        results, info_ = tuple(zip(*self.env.call("reset_done", *args, **kwargs)))
        observations = create_empty_array(
            self.env.single_observation_space, n=self.num_envs, fn=np.empty
        )
        info = {"info_tuple": tuple(info_)}
        return (
            concatenate(self.env.single_observation_space, results, observations),
            info,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info_ = self.env.step(action)
        
        info_["is_success"] = (
                (info_["is_success"] if len(info_["is_success"].shape) == 2 else info_["is_success"].reshape(-1,1))
                if "is_success" in info_
                else np.array([False] * self.num_envs).reshape((self.num_envs, 1))
            )
        
        return (
            obs,
            reward.reshape((self.num_envs, -1)),
            np.array(terminated).reshape((self.num_envs, -1)),
            np.array(truncated).reshape((self.num_envs, -1)),
            info_,
        )


def _worker_shared_memory_no_auto_reset(
    index, env_fn, pipe, parent_pipe, shared_memory, error_queue
):
    """
    This function is derived from _worker_shared_memory() in gym. See:
    https://github.com/openai/gym/blob/master/gym/vector/async_vector_env.py
    """
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, info = env.reset(**data)
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(((None, info), True))
            elif command == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                # NO AUTOMATIC RESET
                # if terminated or truncated:
                #     old_observation = observation
                #     observation, info = env.reset()
                #     info["final_observation"] = old_observation
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(((None, reward, terminated, truncated, info), True))
            # elif command == "seed":
            #     env.seed(data)
            #     pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if name == "reset_done":
                    pipe.send((function(index, *args, **kwargs), True))
                else:
                    if callable(function):
                        pipe.send((function(*args, **kwargs), True))
                    else:
                        pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send(
                    ((data[0] == observation_space, data[1] == env.action_space), True)
                )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
