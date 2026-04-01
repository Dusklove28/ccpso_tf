from env.EnvBase import Env
import numpy as np
from log import logger
from functions import CEC_functions

import random


def fit(x):
    return np.sum(np.power(x, 2))


def sqrt(a, n):
    if a == 0:
        return 0
    b = -1 * (a < 0) + 1 * (a > 0)
    return (a * b) ** (1 / n) * b


class function_wrapper:
    max = 10
    min = -10
    finish = 1e-15

    def __init__(self, dim, fun_num):
        self.cec_functions = CEC_functions(dim)
        self.fun_num = fun_num

    def fun(self, x):
        if len(x.shape) == 2:
            ans = []
            for row in x:
                y = self.cec_functions.Y(row, self.fun_num)
                ans.append(y)
        else:
            ans = self.cec_functions.Y(x, self.fun_num)

        return np.array(ans)


class NormalEnv(Env):
    def __init__(self, obs_shape=(1,), action_shape=(50,), action_low=-1, action_high=1, show=False,
                 target_optimizer=None, fun_nums=None, n_part=100, max_fe=1e4, n_dim=50, group=1):
        super().__init__(obs_shape=obs_shape, action_shape=action_shape, action_low=action_low, action_high=action_high)
        self.target_optimizer = target_optimizer
        self.optimizer = None
        self.fun_nums = fun_nums
        self.fit_value = [0., 0., 0., 0., 0.]

        self.step_num = 0
        self.show_flag = show

        self.min_value = 0

        self.run_time = 0
        self.n_part = n_part
        self.max_fe = max_fe
        self.n_dim = n_dim
        self.group = group

    def reset(self):
        """

        :return: next_state
        """
        n_dim = self.n_dim
        if self.max_fe < self.n_part:
            raise ValueError('max_fe must be at least n_part so the initial population can be evaluated.')
        if self.max_fe % self.n_part != 0:
            raise ValueError('max_fe must be divisible by n_part in the current full-swarm evaluation setup.')
        self.n_run = n_run = int((self.max_fe - self.n_part) / self.n_part)
        show = self.show_flag

        self.fun_num = random.choice(self.fun_nums)

        fun_class = function_wrapper(n_dim, self.fun_num)
        self.optimizer = self.target_optimizer(n_run, self.n_part, show, fun_class.fun, n_dim, 100, -100,
                                               {'max_fes': self.max_fe, 'group': self.group})

        self.fit_value = [0., 0., 0., 0., 0.]
        self.step_num = 0
        self.old_data = {
            'mean': 0,
            'best': 0,
        }

        # next_state, reword, done, _ = self.step(None, init=True)
        return self.optimizer.get_state()

    def test(self):
        done = False
        step_num = 0
        self.reset()
        while not done:
            a, b, done, c = self.step(None, True)
            step_num += 1
        return step_num

    def step(self, action, init=False):
        """
        :param action: 动作
        :return:
        next_state
        reword
        done
        none
        """

        if action is None:
            action = np.zeros(self.action_space.shape[0], dtype=float)
        elif hasattr(action, 'numpy'):
            action = action.numpy()
        else:
            action = np.asarray(action)
        if self.optimizer.fe_num >= self.max_fe or not self.optimizer.run_flag:
            next_state = self.optimizer.get_state()
            return np.array(next_state), 0, True, None
        done = False
        self.step_num += 1

        if self.optimizer.show:
            self.optimizer.show_method()

        self.optimizer.run_once(action)

        # if self.pso_swarm.best_fit < self.fun.finish or self.step_num >= self.n_run:
        #     done = True
        if not self.optimizer.run_flag or self.optimizer.fe_num >= self.max_fe:
            done = True

        num = 0
        # fit = 0
        # for atom in self.pso_swarm.atoms:
        #     num += 1
        #     fit += atom.fitness()
        # mean_fit = fit / num
        # old_mean = self.old_data['mean']
        old_best = self.old_data['best']
        # self.old_data['mean'] = mean_fit
        self.old_data['best'] = self.optimizer.history_best_fit

        deta_best = self.optimizer.history_best_fit - old_best
        # deta_mean = mean_fit - old_mean

        # if not init:
        #     self.fit_value.append(deta_mean)
        #     del self.fit_value[0]

        next_state = self.optimizer.get_state()
        # next_state.append(self.step_num * 0.001)
        # print(f'state:{next_state}\naction:{action[:10]}\nmean:{np.mean(action)},std:{np.std(action)}')

        if deta_best < 0:
            # reward = sqrt(deta_best, 3)
            reward = 1
        else:
            reward = -1

        if np.isnan(reward):
            print(deta_best)
            raise BaseException('reward is None')

        if init:
            reward = 0
        if self.show_flag:
            print('action:{} next_state:{} reward:{} done:{} best:{}'.format(action, next_state, reward, done,
                                                                             self.optimizer.history_best_fit))

        if done:
            # 2. 【核心改造】：计算进度百分比并使用 logger 输出
            fe_progress = min(self.optimizer.fe_num, self.max_fe)
            progress_pct = (fe_progress / self.max_fe) * 100

            # 使用 getattr 安全获取当前算法名称，防止找不到报错
            alg_name = getattr(self.optimizer, 'name', 'Unknown_Alg')

            res = (f"[{alg_name}] Func: {self.fun_num} | "
                   f"FE: {fe_progress}/{self.max_fe} ({progress_pct:.1f}%) | "
                   f"Iter: {self.step_num}/{self.n_run} | "
                   f"Target: {self.min_value} | Result: {self.optimizer.history_best_fit:.4e}")
            logger.info(res)

            # 使用你配置好的 logger！
            logger.info(res)

            # (可选) 保留你原有的 json 写入作为备份
            with open('res2.json', 'a', encoding='utf-8') as f:
                f.write(f'{res}\n')

        return np.array(next_state), reward, done, None
