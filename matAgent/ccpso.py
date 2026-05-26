import numpy as np
from matAgent.baseAgent import MatSwarm



class ConvPsoSwarm(MatSwarm):
    optimizer_name = 'Conv_PSO_DualC'
    action_space = 1
    obs_space = 15

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = self.optimizer_name
        self.fixed_conv_a = self.config.get('fixed_conv_a')
        if self.fixed_conv_a is not None:
            self.fixed_conv_a = float(self.fixed_conv_a)
        self.ccpso_update_mode = self.config.get('ccpso_update_mode', 'second_order')
        self.conv_a_schedule = self.config.get('conv_a_schedule', 'progress_prior')
        self.conv_a_max = float(self.config.get('conv_a_max', 1.5))
        self.conv_a_min = float(self.config.get('conv_a_min', 0.2))
        self.conv_a_delta_scale = float(self.config.get('conv_a_delta_scale', 0.2))
        self.conv_a_clip_min = float(self.config.get('conv_a_clip_min', 0.05))
        self.conv_a_clip_max = float(self.config.get('conv_a_clip_max', 1.8))
        self.stagnation_boost_max = float(self.config.get('stagnation_boost_max', 0.25))
        self.stagnation_boost_fe_ratio = float(self.config.get('stagnation_boost_fe_ratio', 0.2))
        self.first_order_sigma_floor = float(self.config.get('first_order_sigma_floor', 0.001))
        # 追踪收敛系数Conv_a
        self.current_conv_a = None
        self.current_conv_a_base = None
        self.current_conv_a_delta = None
        self.current_conv_a_progress = None
        self.current_stagnation_boost = None
        self.conv_trace = []

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
        # pso.py 的初始化（包含初始速度边界，以保证与对照组的起点绝对公平）
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

        # 新增利用初始速度倒推上一代的假想位置
        self.xs_old = self.xs - self.vs

    def set_x(self, x):
        assert x.shape == self.xs.shape
        self.xs = x

    def update_best(self):
        # pso.py 的最优值更新逻辑
        for i in range(self.n_part):
            if self.fits[i] < self.atom_best_fits[i]:
                self.p_best[i] = self.xs[i].copy()
                self.atom_best_fits[i] = self.fits[i]

        gbest_index = np.argmin(self.fits)
        if self.history_best_fit > self.fits[gbest_index]:
            self.history_best_fit = self.fits[gbest_index]
            self.history_best_x = self.xs[gbest_index].copy()
            self.best_update()

    def _get_progress(self):
        return float(np.clip(self.fe_num / max(self.fe_max, 1), 0.0, 1.0))

    def _get_stagnation_boost(self):
        denominator = max(self.fe_max * self.stagnation_boost_fe_ratio, 1.0)
        no_improve_fe = max(self.fe_num - self.last_best_update_fe, 0)
        stagnation_ratio = np.clip(no_improve_fe / denominator, 0.0, 1.0)
        return float(self.stagnation_boost_max * stagnation_ratio)

    def _resolve_conv_a(self, actions):
        progress = self._get_progress()
        if self.fixed_conv_a is not None:
            conv_a = float(np.clip(self.fixed_conv_a, 0.0, 2.0))
            return conv_a, conv_a, 0.0, progress, 0.0

        if actions is None:
            actions = np.zeros(self.action_space, dtype=float)
        actions = np.asarray(actions, dtype=float).reshape(-1)
        raw_action = float(actions[0])

        if self.conv_a_schedule == 'progress_prior':
            conv_a_base = self.conv_a_max * (1.0 - progress) + self.conv_a_min * progress
            conv_a_delta = raw_action * self.conv_a_delta_scale
            stagnation_boost = self._get_stagnation_boost()
            conv_a = conv_a_base + conv_a_delta + stagnation_boost
        else:
            # 兼容旧实验：Actor 直接输出 Conv_a - 1。
            conv_a_base = 1.0
            conv_a_delta = raw_action
            stagnation_boost = 0.0
            conv_a = raw_action + 1.0

        conv_a = float(np.clip(conv_a, self.conv_a_clip_min, self.conv_a_clip_max))
        return conv_a, float(conv_a_base), float(conv_a_delta), progress, float(stagnation_boost)

    def run_once(self, actions=None):
        Conv_a, Conv_a_base, Conv_a_delta, progress, stagnation_boost = self._resolve_conv_a(actions)
        self.current_conv_a = Conv_a
        self.current_conv_a_base = float(Conv_a_base)
        self.current_conv_a_delta = float(Conv_a_delta)
        self.current_conv_a_progress = float(progress)
        self.current_stagnation_boost = float(stagnation_boost)

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

        if self.ccpso_update_mode == 'second_order':
            # 计算X_Q
            X_Q = a1 * (self.xs - Q) + a2 * (self.xs_old - Q)

            # RL 实施收敛性控制
            new_xs = Q + Conv_a * X_Q
        elif self.ccpso_update_mode == 'first_order':
            sigma = Conv_a * np.abs(Q - self.xs) + self.first_order_sigma_floor
            new_xs = Q + np.random.normal(0.0, sigma, self.xs.shape)
        else:
            raise ValueError(f"unknown ccpso_update_mode: {self.ccpso_update_mode}")

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

        # 保存当前速度（保证其它可能依赖 vs 的接口不报错）
        self.vs = implicit_vs.copy()

        # 计算适应度并更新最优记录
        self.fits = self.fun(self.xs)
        self.update_best()
        self.conv_trace.append((int(self.fe_num), float(self.current_conv_a)))
