import cProfile
import re
import pstats, io
from pstats import SortKey
import os
import matplotlib.pyplot as plt
import numpy as np
from xpag.tools.eval import single_rollout_eval, multiple_rollout_eval
from xpag.tools.utils import get_datatype, datatype_convert, hstack, logical_or
from xpag.tools.logging import eval_log_reset
from xpag.tools.timing import timing_reset
from xpag.buffers import Buffer
from xpag.agents.agent import Agent
from xpag.setters.setter import Setter
from xpag.plotting.plotting import plot_achieved_goals, update_csv
from typing import Dict, Any, Union, List, Optional, Callable


def learn(
    env,
    eval_env,
    env_info: Dict[str, Any],
    agent: Agent,
    buffer: Buffer,
    setter: Setter,
    *,
    batch_size: int = 256,
    gd_steps_per_step: int = 1,
    start_training_after_x_steps: int = 0,
    max_steps: int = 1_000_000_000,
    evaluate_every_x_steps: int = np.inf,
    save_agent_every_x_steps: int = np.inf,
    save_dir: Union[None, str] = None,
    save_episode: bool = False,
    plot_projection: Optional[Callable] = None,
    custom_eval_function: Optional[Callable] = None,
    additional_step_keys: Optional[List[str]] = None,
    seed: Optional[int] = None,
    mult_eval_env = None,
    force_eval_goal_fn = None,
    plot_goals = False,
):
    """
    The function that runs the main training loop.

    It "plays" parallel rollouts, using the agent to choose actions, calling the setter,
    collecting transitions and putting them in the buffer, and training the agent on
    batches sampled from the buffer. It also uses the evaluation environment (eval_env)
    to periodically evaluate the performance of the agent.

    Args:
        env: the environment used for training (multiple rollouts in parallel).
        eval_env: the environment used for evaluation (identical to env except that it
            runs a single rollout).
        env_info: dictionary with information about the env (returned by gym_vec_env()
            and brax_vec_env()).
        agent: the agent.
        buffer: the buffer.
        setter: the setter.
        batch_size (int): the size of the batches of transitions on which the agent is
            trained.
        gd_steps_per_step (int): the number of gradient steps (i.e. calls to
            agent.train_on_batch()) per step in the environment (remark:
            if there n rollouts in parallel, one call to env.step() counts as n steps).
        start_training_after_x_steps (int): the number of inital steps with random
            actions before starting using and training the agent.
        max_steps (int): the maximum number of steps in the environment before stopping
            the learning (remark: if there n rollouts in parallel, one call to
            env.step() counts as n steps).
        evaluate_every_x_steps (int): the number of steps between two evaluations of the
            agent (remark: if there n rollouts in parallel, one call to
            env.step() counts as n steps).
        save_agent_every_x_steps (int): it defines how often the agent is saved to
            the disk (remark: if there n rollouts in parallel, one call to
            env.step() counts as n steps).
        save_dir (str): the directory in which the config, agent, plots, evaluation
            episodes and logs are saved.
        save_episode (bool): if True, the evaluation episodes are saved.
        plot_projection (Callable): a function with 2D outputs from either the
            observation space or the achieved/desired goal space (in the case of a
            goal-based environment). It is used to plot evaluation episodes.
        custom_eval_function (Callable): a custom function used to replace the
            default function for evaluations (single_rollout_eval).
        additional_step_keys (Optional[List[str]]): by default, the transitions are
            stored as dicts with the following entries: "observation", "action",
            "reward", "terminated", "truncated", "next_observation".
            additional_step_keys lists optional additional entries that would be stored
            in the info dict returned by env.step() and setter.step().
        seed (Optional[int]): the random seed for the training.
            Remark: JAX/XLA is not deterministic on GPU, so with JAX agents, the seed
            does not prevent results from varying.
    """
    master_rng = np.random.RandomState(
        seed if seed is not None else np.random.randint(1e9)
    )
    # seed action_space sample
    env_info["action_space"].seed(master_rng.randint(1e9))

    eval_log_reset()
    timing_reset()
    reset_obs, reset_info = env.reset(seed=master_rng.randint(1e9))
    env_datatype = get_datatype(
        reset_obs if not env_info["is_goalenv"] else reset_obs["observation"]
    )
    observation, _ = setter.reset(env, reset_obs, reset_info)

    episodic_buffer = True if hasattr(buffer, "store_done") else False

    if custom_eval_function is None:
        rollout_eval = single_rollout_eval
    else:
        rollout_eval = custom_eval_function

    for i in range(max_steps // env_info["num_envs"]):
        
        if not i % max(evaluate_every_x_steps // env_info["num_envs"], 1):
            rollout_eval(
                i * env_info["num_envs"],
                eval_env,
                env_info,
                agent,
                setter,
                save_dir=save_dir,
                plot_projection=plot_projection,
                save_episode=save_episode,
                env_datatype=env_datatype,
                seed=master_rng.randint(1e9),
            )
            
            if i > 0:
                
                pr.disable()
                s = io.StringIO()
                sortby = SortKey.CUMULATIVE
                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                ps.print_stats(10)
                print(s.getvalue())
            
            
            
            pr = cProfile.Profile()
            

            if mult_eval_env is not None:
                success = multiple_rollout_eval(
                    i * env_info["num_envs"],
                    mult_eval_env,
                    env_info,
                    agent,
                    setter,
                    save_dir=save_dir,
                    env_datatype=env_datatype,
                    seed=master_rng.randint(1e9),
                    goal_fn=force_eval_goal_fn
                )
                
                if plot_projection is not None:
                    
                    # WARNING ! Only valid for sibrivalry custom maze env
                    
                    size = int(eval_env.size_max[0] + 1)
                    success_array = np.array(success).reshape(size**2,-1,1)
                    succ = success_array.mean(axis=1)
                    
                    eval_env.plot()
                    plt.imshow(succ.reshape(size,-1).swapaxes(0,1),origin='lower',extent=(-0.5,size-0.5,-0.5,size-0.5),vmin=0,vmax=1, cmap='RdBu', alpha=0.5)
                    #plt.title("Average Success ({}) : {} %".format(method,str(succ.mean())[:4]))
                    cbar = plt.colorbar()
                    for (v,k),label in np.ndenumerate(succ.reshape(size,-1).swapaxes(0,1)):
                        plt.text(k,v,round(label,2),ha='center',va='center')
                    plt.show()
                    cbar.set_label('proba', rotation=270, labelpad=15)
                    os.makedirs(os.path.join(os.path.expanduser(save_dir), "plots", "grid_eval"), exist_ok=True)
                    steps = i * env_info["num_envs"]
                    grid_eval_path = os.path.join(
                            os.path.expanduser(save_dir),
                            "plots", "grid_eval",
                            f"{steps:12}.png".replace(" ", "0")
                    )
                    plt.savefig(grid_eval_path, bbox_inches='tight')
                    
            
            if env_info['is_goalenv'] and i > 0:
                buffers = buffer.pre_sample()
                episode_max = buffer.current_size
                episode_range = evaluate_every_x_steps // env_info['max_episode_steps']
                last_episode_idxs = np.arange(episode_max - episode_range, episode_max - env_info["num_envs"])
                    
                # Visualisation of achievd and behavior goals
                if plot_goals:
                    plot_achieved_goals(buffers, last_episode_idxs, i*env_info["num_envs"], save_dir)
                intrinsic_success = buffers["is_success"][last_episode_idxs].max(axis=1).mean()
                update_csv("intrinsic_success", intrinsic_success, i*env_info["num_envs"], save_dir)
      

        if not i % max(save_agent_every_x_steps // env_info["num_envs"], 1):
            if save_dir is not None:
                agent.save(os.path.join(os.path.expanduser(save_dir), "agent"))
                setter.save(os.path.join(os.path.expanduser(save_dir), "setter"))


        action_info = {}
        if i * env_info["num_envs"] < start_training_after_x_steps:
            action = env_info["action_space"].sample()
        else:
            action = agent.select_action(
                observation
                if not env_info["is_goalenv"]
                else hstack(observation["observation"], observation["desired_goal"]),
                eval_mode=False,
            )
            if isinstance(action, tuple):
                action_info = action[1]
                action = action[0]
            if i > 0:
                for _ in range(int(max(gd_steps_per_step * env_info["num_envs"], 1))):
                    _ = agent.train_on_batch(buffer.sample(batch_size))

        
        action = datatype_convert(action, env_datatype)
        
        pr.enable()

        next_observation, reward, terminated, truncated, info = setter.step(
            env, observation, action, action_info, *env.step(action)
        )
        
        pr.disable()

        step = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "next_observation": next_observation,
        }
        if env_info["is_goalenv"]:
            step["is_success"] = info["is_success"]
        if additional_step_keys is not None:
            for a_s_key in additional_step_keys:
                if a_s_key in info:
                    step[a_s_key] = info[a_s_key]

        buffer.insert(step)
        observation = next_observation
        
        pr.enable()

        done = logical_or(terminated, truncated)
        if done.max():
            # use store_done() if the buffer is an episodic buffer
            if episodic_buffer:
                buffer.store_done(done)
            observation, _, _ = setter.reset_done(
                env,
                *env.reset_done(done, seed=master_rng.randint(1e9)),
                done,
            )
            
        pr.disable()

 






def learn_change_env(
    env_list: list,
    eval_env_list: list,
    mult_eval_env_list: list,
    change_env_steps_list: list,
    env_info: Dict[str, Any],
    agent: Agent,
    buffer: Buffer,
    setter: Setter,
    *,
    batch_size: int = 256,
    gd_steps_per_step: int = 1,
    start_training_after_x_steps: int = 0,
    max_steps: int = 1_000_000_000,
    evaluate_every_x_steps: int = np.inf,
    save_agent_every_x_steps: int = np.inf,
    save_dir: Union[None, str] = None,
    save_episode: bool = False,
    plot_projection: Optional[Callable] = None,
    custom_eval_function: Optional[Callable] = None,
    additional_step_keys: Optional[List[str]] = None,
    seed: Optional[int] = None,
):
    """
    The function that runs the main training loop whith a change of the environment dynamic.
    This allow us to see the adaptation of the agent to a new dynamic regarding states/goals/action.

    It "plays" parallel rollouts, using the agent to choose actions, calling the setter,
    collecting transitions and putting them in the buffer, and training the agent on
    batches sampled from the buffer. It also uses the evaluation environment (eval_env)
    to periodically evaluate the performance of the agent.

    Args:
        env: the environment used for training (multiple rollouts in parallel).
        eval_env: the environment used for evaluation (identical to env except that it
            runs a single rollout).
        env_info: dictionary with information about the env (returned by gym_vec_env()
            and brax_vec_env()).
        agent: the agent.
        buffer: the buffer.
        setter: the setter.
        batch_size (int): the size of the batches of transitions on which the agent is
            trained.
        gd_steps_per_step (int): the number of gradient steps (i.e. calls to
            agent.train_on_batch()) per step in the environment (remark:
            if there n rollouts in parallel, one call to env.step() counts as n steps).
        start_training_after_x_steps (int): the number of inital steps with random
            actions before starting using and training the agent.
        max_steps (int): the maximum number of steps in the environment before stopping
            the learning (remark: if there n rollouts in parallel, one call to
            env.step() counts as n steps).
        evaluate_every_x_steps (int): the number of steps between two evaluations of the
            agent (remark: if there n rollouts in parallel, one call to
            env.step() counts as n steps).
        save_agent_every_x_steps (int): it defines how often the agent is saved to
            the disk (remark: if there n rollouts in parallel, one call to
            env.step() counts as n steps).
        save_dir (str): the directory in which the config, agent, plots, evaluation
            episodes and logs are saved.
        save_episode (bool): if True, the evaluation episodes are saved.
        plot_projection (Callable): a function with 2D outputs from either the
            observation space or the achieved/desired goal space (in the case of a
            goal-based environment). It is used to plot evaluation episodes.
        custom_eval_function (Callable): a custom function used to replace the
            default function for evaluations (single_rollout_eval).
        additional_step_keys (Optional[List[str]]): by default, the transitions are
            stored as dicts with the following entries: "observation", "action",
            "reward", "terminated", "truncated", "next_observation".
            additional_step_keys lists optional additional entries that would be stored
            in the info dict returned by env.step() and setter.step().
        seed (Optional[int]): the random seed for the training.
            Remark: JAX/XLA is not deterministic on GPU, so with JAX agents, the seed
            does not prevent results from varying.
    """
    
    # init env
    env_idx = 0
    env = env_list[env_idx]
    eval_env = eval_env_list[env_idx]
    mult_eval_env = mult_eval_env_list[env_idx]
    
    master_rng = np.random.RandomState(
        seed if seed is not None else np.random.randint(1e9)
    )
    # seed action_space sample
    env_info["action_space"].seed(master_rng.randint(1e9))

    eval_log_reset()
    timing_reset()
    reset_obs, reset_info = env.reset(seed=master_rng.randint(1e9))
    env_datatype = get_datatype(
        reset_obs if not env_info["is_goalenv"] else reset_obs["observation"]
    )
    observation, _ = setter.reset(env, reset_obs, reset_info)

    episodic_buffer = True if hasattr(buffer, "store_done") else False

    if custom_eval_function is None:
        rollout_eval = single_rollout_eval
    else:
        rollout_eval = custom_eval_function

    for i in range(max_steps // env_info["num_envs"]):
        # Change the environment dynamic at the right step
        if env_idx < len(change_env_steps_list) and i * env_info["num_envs"] > change_env_steps_list[env_idx]:
            
            # End all running episode
            if episodic_buffer:
                buffer.store_done(np.array([[True]*env_info["num_envs"]]).reshape(-1,1))
            
            env_idx+=1
            env = env_list[env_idx]
            eval_env = eval_env_list[env_idx]
            mult_eval_env = mult_eval_env_list[env_idx]
            
            # Init new env
            reset_obs, reset_info = env.reset(seed=master_rng.randint(1e9))
            
            env_datatype = get_datatype(
            reset_obs if not env_info["is_goalenv"] else reset_obs["observation"]
                )   
            observation, _ = setter.reset(env, reset_obs, reset_info)

        
        if not i % max(evaluate_every_x_steps // env_info["num_envs"], 1):
            rollout_eval(
                i * env_info["num_envs"],
                eval_env,
                env_info,
                agent,
                setter,
                save_dir=save_dir,
                plot_projection=plot_projection,
                save_episode=save_episode,
                env_datatype=env_datatype,
                seed=master_rng.randint(1e9),
            )

            multiple_rollout_eval(
                i * env_info["num_envs"],
                mult_eval_env,
                env_info,
                agent,
                setter,
                save_dir=save_dir,
                env_datatype=env_datatype,
                seed=master_rng.randint(1e9),
            )

            if env_info['is_goalenv'] and i > 0:
                buffers = buffer.pre_sample()
                episode_max = buffer.current_size
                episode_range = evaluate_every_x_steps // env_info['max_episode_steps']
                last_episode_idxs = np.arange(episode_max - episode_range, episode_max - env_info["num_envs"])
                    
                # Visualisation of achievd and behavior goals
                #plot_achieved_goals(buffers, last_episode_idxs, i*env_info["num_envs"], save_dir)
                intrinsic_success = buffers["is_success"][last_episode_idxs].max(axis=1).mean()
                update_csv("intrinsic_success", intrinsic_success, i*env_info["num_envs"], save_dir)


        if not i % max(save_agent_every_x_steps // env_info["num_envs"], 1):
            if save_dir is not None:
                agent.save(os.path.join(os.path.expanduser(save_dir), "agent"))
                setter.save(os.path.join(os.path.expanduser(save_dir), "setter"))


        action_info = {}
        if i * env_info["num_envs"] < start_training_after_x_steps:
            action = env_info["action_space"].sample()
        else:
            action = agent.select_action(
                observation
                if not env_info["is_goalenv"]
                else hstack(observation["observation"], observation["desired_goal"]),
                eval_mode=False,
            )
            if isinstance(action, tuple):
                action_info = action[1]
                action = action[0]
            if i > 0:
                for _ in range(int(max(gd_steps_per_step * env_info["num_envs"], 1))):
                    _ = agent.train_on_batch(buffer.sample(batch_size))

        
        action = datatype_convert(action, env_datatype)

        next_observation, reward, terminated, truncated, info = setter.step(
            env, observation, action, action_info, *env.step(action)
        )

        step = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "next_observation": next_observation,
        }
        if env_info["is_goalenv"]:
            step["is_success"] = info["is_success"]
        if additional_step_keys is not None:
            for a_s_key in additional_step_keys:
                if a_s_key in info:
                    step[a_s_key] = info[a_s_key]

        buffer.insert(step)
        observation = next_observation

        done = logical_or(terminated, truncated)
        
        if done.max():
            # use store_done() if the buffer is an episodic buffer
            if episodic_buffer:
                buffer.store_done(done)
            observation, _, _ = setter.reset_done(
                env,
                *env.reset_done(done, seed=master_rng.randint(1e9)),
                done,
            )