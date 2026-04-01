import numpy as np
from matAgent.ccpso import FiftyDimCCPsoSwarm

class RlCCPsoSwarm(FiftyDimCCPsoSwarm):
    optimizer_name = 'RL_CCPSO50D'

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        config_dic = {} if config_dic is None else dict(config_dic)
        model_path = config_dic.get('model')
        if not model_path:
            raise ValueError("评估 RL-CCPso50D 必须提供 model 路径")

        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'RL-CCPso50D'
        self.optimizer_name = self.name

    # 【修复点】：action -> actions, 加上 **kwargs
    def run_once(self, actions=None, **kwargs):
        state = self.get_state()
        my_action = self.ddpg_actor.policy(state)
        # 向下传递时，使用父类规定的 actions
        super().run_once(actions=my_action.numpy())