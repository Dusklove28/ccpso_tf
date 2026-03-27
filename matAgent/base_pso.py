import numpy as np
from matAgent.baseAgent import MatSwarm


class BasePsoSwarm(MatSwarm):
    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'CC_PSO'

        # 算法所需的状态变量
        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)
        self.atom_history_best_fits = np.zeros(self.n_part)
        self.g_best = np.zeros(n_dim)
        self.g_best_index = 0
        self.fits = np.zeros(self.n_part)

        # 统一式特有的变量：记录上一代的位置 X(t-1)
        self.xs_old = np.zeros_like(self.xs)

        # 评价次数记录（用于计算进度）
        self.fe_max = config_dic.get('max_fes', 20000) if config_dic else 20000
        self.fe_num = 0

        self.init()

    def init(self):
        # 随机初始化位置和速度
        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.vs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.xs_old = self.xs.copy()  # 第一代时，X(t-1) 就等于 X(t)

        self.fits = self.fun(self.xs)
        self.fe_num += self.n_part
        self.init_finish = True

        # 初始化历史最优
        self.g_best_index = np.argmin(self.fits)
        self.history_best_fit = self.fits[self.g_best_index]
        self.g_best = self.xs[self.g_best_index].copy()
        self.atom_history_best_fits = self.fits.copy()
        self.p_best = self.xs.copy()

    def update_best(self):
        # 更新个体最优 Pbest
        for i in range(self.n_part):
            if self.fits[i] < self.atom_history_best_fits[i]:
                self.p_best[i] = self.xs[i].copy()
                self.atom_history_best_fits[i] = self.fits[i]

        # 更新全局最优 Gbest
        self.g_best_index = np.argmin(self.fits)
        if self.history_best_fit > self.fits[self.g_best_index]:
            self.history_best_fit = self.fits[self.g_best_index]
            self.g_best = self.xs[self.g_best_index].copy()
            self.best_update()

    def run_once(self, action):
        # 1. 解析动作并映射到合理范围（DDPG网络输出的 action 范围固定是 [-1, 1]）
        #    如果 action[0] 是 -1，w = 0.1；如果是 1，w = 0.9。下同。
        w = action[0] * 0.4 + 0.5  # 将 [-1, 1] 映射到惯性权重 [0.1, 0.9]
        c1 = action[1] * 1.0 + 1.5  # 将 [-1, 1] 映射到认知因子 [0.5, 2.5]
        c2 = action[2] * 1.0 + 1.5  # 将 [-1, 1] 映射到社会因子 [0.5, 2.5]
        P_ECon = action[3] * 0.5 + 0.5  # 将 [-1, 1] 映射到期望收敛度 [0.0, 1.0]

        r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))

        c1_r1 = c1 * r1
        c2_r2 = c2 * r2
        C = c1_r1 + c2_r2

        # 2. 计算等效吸引中心 Q (为避免除零加上 1e-16)
        Q = (c1_r1 * self.p_best + c2_r2 * self.g_best) / (C + 1e-16)

        # 3. 计算统一式系数 a1 和 a2
        a1 = 1 + w - C
        a2 = -w

        # 4. 计算统一式的期望位置偏差 X_Q
        X_Q = a1 * (self.xs - Q) + a2 * (self.xs_old - Q)

        # 5. 计算实际收敛度 P_Con (这里以各粒子分开的 1-范数 为例)
        P_Con = np.linalg.norm(X_Q, ord=1, axis=1, keepdims=True)
        P_Con[P_Con == 0] = 1e-16  # 避免除以0

        # 6. 应用收敛性控制公式更新位置
        new_xs = Q + (P_ECon / P_Con) * X_Q

        # 7. 越界处理
        new_xs = np.clip(new_xs, self.pos_min, self.pos_max)

        # 8. 状态迭代更替
        self.xs_old = self.xs.copy()  # 当前位置变成老位置 X(t-1)
        self.xs = new_xs  # 更新当前位置 X(t)

        # 9. 重新评估适应度并更新 Best
        self.fits = self.fun(self.xs)
        self.fe_num += self.n_part
        self.update_best()

    def get_state(self):
        # 兼容原环境：返回训练进度作为 RL 的 State 观察值
        return [(self.fe_num / self.fe_max - 0.5) * 2]