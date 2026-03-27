from pathlib import Path

import numpy as np

from matAgent.testpso import TestpsoSwarm


class RlepsoSwarm(TestpsoSwarm):
    optimizer_name = 'RLEPSO'

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        config_dic = {} if config_dic is None else dict(config_dic)
        config_dic.setdefault(
            'model',
            r"D:\develop\swam\ieeeaccess - 副本 - 副本 - 副本 - 副本\rl\train0\ddpg_actor_episode100.h5"
        )

        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)

        model_name = Path(config_dic['model']).name
        self.name = f'RLEPSO-{model_name}'
        self.optimizer_name = self.name

   # 【修复点】：action -> actions, 加上 **kwargs
    def run_once(self, actions=None, **kwargs):
        state = self.get_state()
        my_action = self.ddpg_actor.policy(state)
        if self.show:
            print(np.mean(np.abs(my_action)))
            print(my_action[:10])
        # 向下传递时，使用父类规定的 actions
        super().run_once(actions=my_action.numpy())


def fun2(x):
    x2 = np.power(x - 50, 2)
    fit = np.sum(x2, axis=-1)
    return fit


if __name__ == '__main__':
    model = r'''D:\develop\swam\ieeeaccess - 副本 - 副本 - 副本 - 副本\rl\train0\ddpg_actor_episode300.h5'''

    s = RlepsoSwarm(1000, 40, True, fun2, 2, 100, -100, {'max_fes': 10000, 'model': model})
    s.run()
