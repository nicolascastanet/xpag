# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import numpy as np
from typing import List, Dict, Any
from xpag.tools.utils import DataType, datatype_convert
import os
import time
import csv


def _from_1d_to_2d(t, v):
    assert len(v) == 1 or len(v) == 2, "projection function outputs must be 1D or 2D"
    if len(v) == 2:
        return 2, v
    else:
        return 1, np.array([t, v[0]])


def _expand_bounds(bounds):
    bmin = bounds[0]
    bmax = bounds[1]
    expand_ratio = 1e-1
    min_expansion = 1e-3
    delta = max((bmax - bmin) * expand_ratio, min_expansion)
    return [bmin - delta, bmax + delta]


def single_episode_plot(
    filename: str,
    step_list: List[Dict[str, Any]],
    projection_function=lambda x: x[0:2],
    plot_env_function=None,
):
    """Plots an episode, using a 1D or 2D projection from observations, or
    from achieved and desired goals in the case of GoalEnv environments.
    """
    from matplotlib import figure  # lazy import
    from matplotlib import collections as mc  # lazy import

    fig = figure.Figure()
    ax = fig.subplots(1)
    xmax = -np.inf
    xmin = np.inf
    ymax = -np.inf
    ymin = np.inf
    lines = []
    obs_list = []
    rgbs = []
    gx = []
    gy = []
    episode_length = len(step_list)
    goalenv = False
    projection_dimension = None
    for t, step in enumerate(step_list):
        if (
            isinstance(step["observation"], dict)
            and "achieved_goal" in step["observation"]
        ):
            goalenv = True
            projection_dimension, x_obs = _from_1d_to_2d(
                t,
                projection_function(
                    datatype_convert(
                        step["observation"]["achieved_goal"][0], DataType.NUMPY
                    )
                ),
            )
            projection_dimension, x_obs_next = _from_1d_to_2d(
                t + 1,
                projection_function(
                    datatype_convert(
                        step["next_observation"]["achieved_goal"][0], DataType.NUMPY
                    )
                ),
            )
            projection_dimension, gxy = _from_1d_to_2d(
                t + 1,
                projection_function(
                    datatype_convert(
                        step["observation"]["desired_goal"][0], DataType.NUMPY
                    )
                ),
            )
            gx.append(gxy[0])
            xmax = max(xmax, gxy[0])
            xmin = min(xmin, gxy[0])
            gy.append(gxy[1])
            ymax = max(ymax, gxy[1])
            ymin = min(ymin, gxy[1])
        else:
            projection_dimension, x_obs = _from_1d_to_2d(
                t,
                projection_function(
                    datatype_convert(step["observation"][0], DataType.NUMPY)
                ),
            )
            projection_dimension, x_obs_next = _from_1d_to_2d(
                t + 1,
                projection_function(
                    datatype_convert(step["next_observation"][0], DataType.NUMPY)
                ),
            )

        obs_list.append(x_obs_next)
        lines.append((x_obs, x_obs_next))
        xmax = max(xmax, max(x_obs[0], x_obs_next[0]))
        xmin = min(xmin, min(x_obs[0], x_obs_next[0]))
        ymax = max(ymax, max(x_obs[1], x_obs_next[1]))
        ymin = min(ymin, min(x_obs[1], x_obs_next[1]))
        rgbs.append(
            (1.0 - t / episode_length / 2.0, 0.2, 0.2 + t / episode_length / 2.0, 1)
        )
    ax.set_xlim(_expand_bounds([xmin, xmax]))
    ax.set_ylim(_expand_bounds([ymin, ymax]))
    if plot_env_function is not None and projection_dimension == 2:
        plot_env_function(ax)
    if goalenv:
        if projection_dimension == 2:
            ax.scatter(gx, gy, s=12, c="green", alpha=0.8)
        else:
            g_gather = np.vstack((gx, gy)).transpose()
            g_lines = list(zip(g_gather[:-1], g_gather[1:]))
            ax.add_collection(
                mc.LineCollection(g_lines, colors="green", linewidths=2.0)
            )

    obs_list = np.array(obs_list)

    ax.scatter(obs_list[:,0], obs_list[:,1], s=10, c="blue")
    ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=1.0))
    fig.savefig(filename, dpi=200)

    #fig.clf()
    #ax.cla()



def add_np_embedding(
    step: int, 
    save_dir, 
    tag: str, 
    values, 
    upper_tag='goals'
):
    
    import os
    if isinstance(values, list):
      values = np.array(values, dtype=np.float32)
    elif isinstance(values, np.ndarray):
      values = values.astype(np.float32)
    assert len(values.shape) == 2
    
    path = os.path.join(os.path.expanduser(save_dir), upper_tag,str(step),tag)
    os.makedirs(path, exist_ok=True)
    np.save(path+'/array.npy',values)



def plot_achieved_goals(
    buffers, 
    episode_idxs, 
    step: int, 
    save_dir
):

    t_max_episodes = buffers["episode_length"][episode_idxs, 0]
    t_max_episodes = t_max_episodes.flatten().astype(int)

    goals = {
        key: buffers[key][episode_idxs, t_max_episodes-1] for key in ["observation.achieved_goal",
                                                                "observation.desired_goal"]
        }

    add_np_embedding(step, save_dir, "ags", goals["observation.achieved_goal"])
    add_np_embedding(step, save_dir, "bgs", goals["observation.desired_goal"])


def update_csv(
    tag: str, 
    value: float, 
    step: int, 
    save_dir
):

    fields = ['step', tag]
    path = os.path.join(save_dir, tag.replace('/', '__') + '.csv')
    if not os.path.exists(path):
      with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(fields)
    with open(path, 'a') as f:
      writer = csv.writer(f)
      writer.writerow([step, value])