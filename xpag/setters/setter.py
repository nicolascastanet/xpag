# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC, abstractmethod
from typing import Tuple, Any, Iterable
from types import MethodType
import os
import numpy as np

# SVGG import
from xpag.svgg.success_prediction import GoalSuccessPredictor
from xpag.svgg.svgd import RBF, SVGD



class Setter(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def reset(self, env, observation, info, eval_mode=False) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def reset_done(
        self, env, observation, info, done, eval_mode=False
    ) -> Tuple[Any, Any, Any]:
        pass

    @abstractmethod
    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode: bool = False,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        pass

    @abstractmethod
    def write_config(self, output_file: str):
        pass

    @abstractmethod
    def save(self, directory: str):
        pass

    @abstractmethod
    def load(self, directory: str):
        pass



class DefaultSetter(Setter, ABC):
    def __init__(self):
        super().__init__("DefaultSetter")

    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):

        return new_observation, reward, terminated, truncated, info

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass



def __getattr__(self, name):
            return getattr(self.setter, name)


class SetterSharedAttr(Setter, ABC):
    """
    __getattr__ passes through to the agent, so that a call to self.* is the same as a
    call to self.setter.* whenever * is not defined. 
    """

    def __init__(self, name: str, modules: Iterable):
        super.__init__(name)

        self.opt_steps = 0
        self._optimize_registry = []

        for m in modules:
            assert m.name
            setattr(self, m.name, m)
            m.setter = self
            if hasattr(m, '_optimize'):
                self._optimize_registry.append(m)
                m.__getattr__ = MethodType(__getattr__, m, type(m))

    def optimize(self):
        """Calls the _optimize function of each relevant module
        (typically, this will be the main algorithm; but may include others)"""
        self.opt_steps += 1
        for module in self._optimize_registry:
            module._optimize()







class UniformSetter(Setter, ABC):
    def __init__(self):
        super().__init__("UniformSetter")

    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        
        goal = np.where(done == 1, env.sample_uniform_goal(), env.goal)
        env.set_goal(np.copy(goal))
        observation["desired_goal"] = np.copy(goal)

        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):

        return new_observation, reward, terminated, truncated, info

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass


class SvggSetter(SetterSharedAttr, ABC):
    def __init__(self, num_envs, criterion, svgd, success_pred, model):
        super().__init__("SvggSetter", modules = [criterion, svgd, success_pred, model])
        self.num_envs = num_envs
        self.criterion = criterion
        self.svgd = svgd


        self.steps = 0


    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        
        goal = np.where(done == 1, env.sample_uniform_goal(), env.goal)
        env.set_goal(np.copy(goal))
        observation["desired_goal"] = np.copy(goal)

        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):

        self.steps += self.num_envs * (1-eval_mode)
        self.optimize()

        return new_observation, reward, terminated, truncated, info

    def optimize(self):
        #if not i % max(save_agent_every_x_steps // env_info["num_envs"], 1)

        # TO DO
        pass

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass


class DensitySetter(Setter, ABC):
    def __init__(self, num_envs, criterion, svgd):
        super().__init__("SvggSetter")
        self.num_envs = num_envs
        self.criterion = criterion
        self.svgd = svgd
        self.steps = 0


    def reset(self, env, observation, info, eval_mode=False):
        return observation, info

    def reset_done(self, env, observation, info, done, eval_mode=False):
        
        goal = np.where(done == 1, env.sample_uniform_goal(), env.goal)
        env.set_goal(np.copy(goal))
        observation["desired_goal"] = np.copy(goal)

        return observation, info, done

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):

        self.steps += self.num_envs * (1-eval_mode)
        self.optimize()

        return new_observation, reward, terminated, truncated, info

    def optimize(self):
        #if not i % max(save_agent_every_x_steps // env_info["num_envs"], 1)

        # TO DO
        pass

    def write_config(self, output_file: str):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass




class CompositeSetter(Setter, ABC):
    def __init__(self, setter1: Setter, setter2: Setter):
        super().__init__("CompositeSetter")
        self.setter1 = setter1
        self.setter2 = setter2

    def reset(self, env, observation, info, eval_mode=False):
        obs_, info_ = self.setter1.reset(env, observation, info, eval_mode)
        return self.setter2.reset(env, obs_, info_, eval_mode)

    def reset_done(self, env, observation, info, done, eval_mode=False):
        obs_, info_, done_ = self.setter1.reset_done(
            env, observation, info, done, eval_mode
        )
        return self.setter2.reset_done(env, obs_, info_, done_, eval_mode)

    def step(
        self,
        env,
        observation,
        action,
        action_info,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
        eval_mode=False,
    ):
        new_obs_, reward_, terminated_, truncated_, info_ = self.setter1.step(
            env,
            observation,
            action,
            action_info,
            new_observation,
            reward,
            terminated,
            truncated,
            info,
            eval_mode,
        )
        return self.setter2.step(
            env,
            observation,
            action,
            action_info,
            new_obs_,
            reward_,
            terminated_,
            truncated_,
            info_,
            eval_mode,
        )

    def write_config(self, output_file: str):
        self.setter1.write_config(output_file + ".1")
        self.setter2.write_config(output_file + ".2")

    def save(self, directory: str):
        self.setter1.save(os.path.join(directory, "1"))
        self.setter2.save(os.path.join(directory, "2"))

    def load(self, directory: str):
        self.setter1.load(os.path.join(directory, "1"))
        self.setter2.load(os.path.join(directory, "2"))
