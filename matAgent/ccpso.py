import numpy as np
from matAgent.baseAgent import MatSwarm



class ConvPsoSwarm(MatSwarm):
    optimizer_name = 'Conv_PSO'
    action_space = 1
    obs_space = 15

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = 'Conv_PSO'
        # 追踪收敛系数Conv_a
        self.current_conv_a = None
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
        self.vs = np.clip(self.vs, self.min_v, self.max_v)
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
        self.xs_old = np.clip(self.xs - self.vs, self.pos_min, self.pos_max)

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
        if self.fe_max <= 0:
            return 0.0
        return float(np.clip(self.fe_num / self.fe_max, 0.0, 1.0))

    def _get_diversity_ratio(self):
        search_span = float(np.max(np.asarray(self.pos_max) - np.asarray(self.pos_min)))
        search_span = max(search_span, 1e-12)
        diversity = float(np.mean(np.std(self.xs, axis=0)))
        return float(np.clip(diversity / search_span, 0.0, 1.0))

    def _get_stagnation_ratio(self):
        if self.fe_max <= 0:
            return 0.0
        no_improve = max(self.fe_num - self.last_best_update_fe, 0)
        return float(np.clip(no_improve / self.fe_max, 0.0, 1.0))

    def _shape_conv(self, raw_conv):
        progress = self._get_progress()
        stagnation = self._get_stagnation_ratio()
        diversity = self._get_diversity_ratio()

        # Keep expansion softer than the original direct scaling to avoid early overshoot.
        if raw_conv >= 1.0:
            shaped_conv = 1.0 + 0.45 * (raw_conv - 1.0)
        else:
            shaped_conv = 1.0 - 0.85 * (1.0 - raw_conv)

        # Favor mild expansion early and mild contraction late; stagnation can reopen exploration.
        phase_conv = 1.05 - 0.35 * progress + 0.20 * stagnation
        if diversity < 0.08:
            phase_conv += 0.10 * (0.08 - diversity) / 0.08

        effective_conv = 0.60 * shaped_conv + 0.40 * phase_conv
        return float(np.clip(effective_conv, 0.45, 1.35))

    def _get_conv_mix(self):
        progress = self._get_progress()
        stagnation = self._get_stagnation_ratio()
        diversity = self._get_diversity_ratio()
        collapse = max(0.10 - diversity, 0.0) / 0.10

        # Early search and stagnation use more of the convergence strategy.
        conv_mix = 0.30 + 0.35 * (1.0 - progress) + 0.25 * stagnation + 0.10 * collapse
        return float(np.clip(conv_mix, 0.25, 0.85))

    def run_once(self, actions=None):
        # 提取 RL 动作 (action_space = 1)
        if actions is None:
            actions = np.zeros(self.action_space, dtype=float)

        actions = np.asarray(actions, dtype=float).reshape(-1)
        # RL只控制收敛性参数 Conv_a 映射到 [0.0, 2.0]
        raw_conv = float(np.clip(actions[0] + 1.0, 0.0, 2.0))
        conv_a = self._shape_conv(raw_conv)
        self.current_conv_a = conv_a


        # 生成与 pso.py 完全一致的随机张量 (n_part, n_dim)
        self.r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        self.r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))

        # 你的策略：固定 Clerc 收缩参数
        conv_w = 0.729844
        conv_c1 = 1.496180
        conv_c2 = 1.496180

        # === 以下为利用 Numpy 广播机制的无 For 循环加速计算 ===
        c1_r1 = conv_c1 * self.r1
        c2_r2 = conv_c2 * self.r2
        C_gravity = c1_r1 + c2_r2

        # 计算等效引力中心 Q（注意：对齐 pso.py，全局最优变量名为 history_best_x）
        Q = (c1_r1 * self.p_best + c2_r2 * self.history_best_x) / (C_gravity + 1e-16)

        # 构建二阶差分系数
        a1 = 1.0 + conv_w - C_gravity
        a2 = -conv_w

        # 计算X_Q
        X_Q = a1 * (self.xs - Q) + a2 * (self.xs_old - Q)

        # RL 实施收敛性控制
        conv_vs = Q + conv_a * X_Q - self.xs

        # 【核心修正】隐式速度截断！
        # 算出假设的速度，并像 pso.py 那样严格进行边界截断，防止失去对比公平性
        pso_w = 0.5
        pso_c1 = 2.0
        pso_c2 = 2.0
        pso_vs = (
            pso_w * self.vs
            + pso_c1 * self.r1 * (self.p_best - self.xs)
            + pso_c2 * self.r2 * (self.history_best_x - self.xs)
        )

        conv_mix = self._get_conv_mix()
        implicit_vs = (1.0 - conv_mix) * pso_vs + conv_mix * conv_vs

        progress = self._get_progress()
        if progress > 0.5:
            late_pull = 0.15 * (progress - 0.5) / 0.5
            implicit_vs += late_pull * (Q - self.xs)
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
