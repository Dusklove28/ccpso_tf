import numpy as np
from matAgent.baseAgent import MatSwarm


class ConvPsoSwarm(MatSwarm):
    optimizer_name = 'Conv_PSO'
    action_space = 1
    obs_space = 15

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'Conv_PSO'

        # 完全复刻 pso.py 的变量结构
        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)
        self.atom_best_fits = np.zeros(self.n_part)
        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part)

        # 矩阵化存储随机因子
        self.r1 = np.zeros((self.n_part, self.n_dim))
        self.r2 = np.zeros((self.n_part, self.n_dim))

        self.init()

    def init(self):
        # 100% 还原 pso.py 的初始化（包含初始速度边界，以保证与对照组的起点绝对公平）
        self.xs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.vs = np.random.uniform(self.pos_min, self.pos_max, self.xs.shape)
        self.fits = self.fun(self.xs)

        gbest_index = np.argmin(self.fits)
        self.history_best_fit = self.fits[gbest_index]
        self.history_best_x = self.xs[gbest_index].copy()
        self.atom_best_fits = self.fits.copy()
        self.p_best = self.xs.copy()
        self.init_finish = True
        self.fe_num = self.n_part
        self.run_flag = self.fe_num < self.fe_max
        if (self.fe_num % self.record_per_fe == 0 or self.fe_num == self.fe_max) and self.fe_num <= self.fe_max:
            self.data_collect_method()

        # 唯一新增变量：为了二阶动力学推导，利用初始速度倒推上一代的假想位置
        self.xs_old = self.xs - self.vs

    def set_x(self, x):
        assert x.shape == self.xs.shape
        self.xs = x

    def update_best(self):
        # 100% 还原 pso.py 的最优值更新逻辑
        for i in range(self.n_part):
            if self.fits[i] < self.atom_best_fits[i]:
                self.p_best[i] = self.xs[i].copy()
                self.atom_best_fits[i] = self.fits[i]

        gbest_index = np.argmin(self.fits)
        if self.history_best_fit > self.fits[gbest_index]:
            self.history_best_fit = self.fits[gbest_index]
            self.history_best_x = self.xs[gbest_index].copy()
            self.best_update()

    def run_once(self, actions=None):
        # 提取 RL 动作 (action_space = 1)
        if actions is None:
            actions = np.zeros(self.action_space)

        # RL只控制收敛性参数 Conv_a 映射到 [0.0, 2.0]
        Conv_a = actions[0] + 1.0

        # 生成与 pso.py 完全一致的随机张量 (n_part, n_dim)
        self.r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        self.r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))

        # 你的策略：固定 Clerc 收缩参数
        w = 0.729844
        c1 = 1.496180
        c2 = 1.496180

        # === 以下为利用 Numpy 广播机制的无 For 循环加速计算 ===
        c1_r1 = c1 * self.r1
        c2_r2 = c2 * self.r2
        C_gravity = c1_r1 + c2_r2

        # 计算等效引力中心 Q（注意：对齐 pso.py，全局最优变量名为 history_best_x）
        Q = (c1_r1 * self.p_best + c2_r2 * self.history_best_x) / (C_gravity + 1e-16)

        # 构建二阶差分系数
        a1 = 1 + w - C_gravity
        a2 = -w

        # 计算底座自然偏移量 X_Q
        X_Q = a1 * (self.xs - Q) + a2 * (self.xs_old - Q)

        # RL 干预项：实施收敛性控制
        new_xs = Q + Conv_a * X_Q

        # 【核心修正】隐式速度截断！
        # 算出假设的速度，并像 pso.py 那样严格进行边界截断，防止失去对比公平性
        implicit_vs = new_xs - self.xs
        implicit_vs = np.clip(implicit_vs, self.min_v, self.max_v)

        # 利用截断后的速度计算真实的新位置
        new_xs = self.xs + implicit_vs
        new_xs = np.clip(new_xs, self.pos_min, self.pos_max)

        # 迭代状态更新
        self.xs_old = self.xs.copy()
        self.xs = new_xs.copy()

        # 隐式保存当前速度（保证其它可能依赖 vs 的接口不报错）
        self.vs = implicit_vs.copy()

        # 计算适应度并更新最优记录
        self.fits = self.fun(self.xs)
        self.update_best()
