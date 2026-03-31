##########固定参数策略+收敛性策略###########
import numpy as np
from matAgent.testpso import TestpsoSwarm  

class FiftyDimCCPsoSwarm(TestpsoSwarm):
    optimizer_name = 'CCPSO_50D'
    action_space = 10
    obs_space = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = '50D_CC_PSO'
        # 记录上一代的位置 X(t-1)
        self.xs_old = self.xs.copy()

    def run_once(self, actions):
        if actions is None:
            actions = np.zeros(self.action_space * self.n_group)
            
        if self.show:
            print('{}|best fit:{}'.format(self.fe_num / self.fe_max, self.history_best_fit))

        new_xs = np.zeros_like(self.xs)

        # 固定w,c1,c2
        # c1 = 1.49445
        # c2 = 1.49445
        # 这里为了配合动态寻优，我们采用标准的线性衰减 [0.9 -> 0.4]
        # w_fixed = 0.9 - 0.5 * (self.fe_num / self.fe_max)

        # 与作者保持一致
        c1 = 2
        c2 = 2
        w = 0.5


        for i in range(self.n_part):
            # 提取动作
            action = actions[
                i % self.n_group * self.action_space:
                i % self.n_group * self.action_space + self.action_space
            ]

            # 【核心修正 2：遵循老师指导，RL 只输出并控制唯一参数 Conv_a】
            # 映射到 [0.1, 1.5] 的区间。
            # > 1 用于前期打破局部最优进行广度探索；< 1 用于后期精细收缩。
            Conv_a = action[0] * 0.7 + 0.8  
            

            for d in range(self.n_dim):
                # 为每个粒子的每个维度生成独立的均匀分布随机数
                r1 = np.random.rand()
                r2 = np.random.rand()

                pbest_target = self.p_best[i, d]
                gbest_target = self.g_best[d]

                c1_r1 = c1 * r1
                c2_r2 = c2 * r2

                C_gravity = c1_r1 + c2_r2  
                
                # 计算严谨的等效引力中心 Q
                Q = (c1_r1 * pbest_target + c2_r2 * gbest_target) / (C_gravity + 1e-16)

                # 构建二阶差分方程的系数
                a1 = 1 + w - C_gravity
                a2 = -w

                # 计算基于稳定底座的自然偏移量 X_Q
                X_Q = a1 * (self.xs[i, d] - Q) + a2 * (self.xs_old[i, d] - Q)

                # 唯一由强化学习干预的环节：缩放位移，实施收敛性控制
                new_xs[i, d] = Q + Conv_a * X_Q

        # 越界处理 (强行拉回搜索空间)
        new_xs = np.clip(new_xs, self.pos_min, self.pos_max)

        # 状态迭代：更新历史轨迹
        self.xs_old = self.xs.copy() 
        self.xs = new_xs             

        # 计算适应度并更新最优记录
        self.fits = self.fun(self.xs)
        self.update_best()       
