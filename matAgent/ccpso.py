import numpy as np

from matAgent.baseAgent import MatSwarm


class ConvPsoSwarm(MatSwarm):
    """DDPG 控制的二阶 CCPSO 群体。

    当前主线只保留一种控制方式：Actor 输出 `Conv_a` 的残差，基础值由
    progress prior 给出。`Conv_a` 只控制围绕等效引力中心 Q 的运动半径，
    不直接改变 Q、pbest 或 gbest。
    """

    optimizer_name = 'Conv_PSO_DualC'
    action_space = 1
    obs_space = 15

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        """初始化 CCPSO 群体、Conv_a 控制参数和诊断变量。"""
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = self.optimizer_name

        # 固定 Conv_a 用于消融；一旦设置，会覆盖 Actor 输出和进度先验。
        self.fixed_conv_a = self.config.get('fixed_conv_a')
        if self.fixed_conv_a is not None:
            self.fixed_conv_a = float(self.fixed_conv_a)

        # Actor 只学习残差，避免直接学习完整的二阶收敛控制律。
        self.conv_a_delta_scale = float(self.config.get('conv_a_delta_scale', 0.2))
        self.conv_a_clip_min = float(self.config.get('conv_a_clip_min', 0.00))
        self.conv_a_clip_max = float(self.config.get('conv_a_clip_max', 2.0))

        # 停滞时轻微提高 Conv_a，给粒子额外探索半径。
        self.stagnation_boost_max = float(self.config.get('stagnation_boost_max', 0.25))
        self.stagnation_boost_fe_ratio = float(self.config.get('stagnation_boost_fe_ratio', 0.2))

        # 以下变量供 evaluate/q_collapse_diagnosis.py 和画图诊断使用。
        self.current_conv_a = None
        self.current_conv_a_base = None
        self.current_conv_a_delta = None
        self.current_conv_a_progress = None
        self.current_stagnation_boost = None
        self.current_q = None
        self.current_pbest = None
        self.current_pbest_fit = None
        self.current_gbest = None
        self.current_gbest_fit = None
        self.current_x_before_update = None
        self.current_gbest_before_update = None
        self.current_pbest_diversity = None
        self.current_c1 = None
        self.current_c2 = None
        self.conv_trace = []

        # 与 PsoSwarm 保持一致的群体状态变量。
        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)
        self.atom_best_fits = np.zeros(self.n_part)
        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part)

        # 每代更新时重新采样的随机因子。
        self.r1 = np.zeros((self.n_part, self.n_dim))
        self.r2 = np.zeros((self.n_part, self.n_dim))

        self.init()

    def init(self):
        """随机初始化粒子位置、速度、pbest/gbest，以及二阶项所需的上一代位置。"""
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

        # 二阶 DualC 需要 x(t-1)。初始化时用 x(t)-v(t) 倒推上一代位置。
        self.xs_old = self.xs - self.vs

    def set_x(self, x):
        """外部调试接口：直接覆盖粒子当前位置。"""
        assert x.shape == self.xs.shape
        self.xs = x

    def update_best(self):
        """根据当前适应度更新每个粒子的 pbest 和整个群体的 gbest。"""
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
        """返回当前 FE 进度，范围裁剪到 [0, 1]。"""
        return float(np.clip(self.fe_num / max(self.fe_max, 1), 0.0, 1.0))

    def _get_stagnation_boost(self):
        """根据距离上次 gbest 改善的 FE 数，计算停滞补偿项。"""
        denominator = max(self.fe_max * self.stagnation_boost_fe_ratio, 1.0)
        no_improve_fe = max(self.fe_num - self.last_best_update_fe, 0)
        stagnation_ratio = np.clip(no_improve_fe / denominator, 0.0, 1.0)
        return float(self.stagnation_boost_max * stagnation_ratio)

    def _normalized_point_diversity(self, points):
        """计算点集按搜索空间范围归一化后的平均坐标标准差。"""
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return 0.0
        search_span = max(float(self.pos_max - self.pos_min), 1e-12)
        return float(np.mean(np.std(points, axis=0)) / search_span)

    def _progress_prior_base(self, progress):
        """分段进度先验：前 60% FE 缓慢下降，后 40% FE 快速进入收敛。"""
        if progress <= 0.6:
            return 1.5 - 0.48 * (progress / 0.6) ** 1.2
        return 1.0 - 0.8 * ((progress - 0.6) / 0.4) ** 0.7

    def _resolve_conv_a(self, actions):
        """把 Actor 动作、进度先验和停滞补偿合成为最终 Conv_a。"""
        progress = self._get_progress()
        if self.fixed_conv_a is not None:
            conv_a = float(np.clip(self.fixed_conv_a, self.conv_a_clip_min, self.conv_a_clip_max))
            return conv_a, conv_a, 0.0, progress, 0.0

        # 训练开始或诊断无模型时，默认 Actor 残差为 0。
        if actions is None:
            actions = np.zeros(self.action_space, dtype=float)
        elif hasattr(actions, 'numpy'):
            actions = actions.numpy()

        actions = np.asarray(actions, dtype=float).reshape(-1)
        raw_action = float(actions[0]) if actions.size else 0.0

        conv_a_base = self._progress_prior_base(progress)
        conv_a_delta = raw_action * self.conv_a_delta_scale
        stagnation_boost = self._get_stagnation_boost()

        # 最终 Conv_a = 人工进度先验 + Actor 残差 + 停滞补偿，再统一裁剪。
        conv_a = conv_a_base + conv_a_delta + stagnation_boost
        conv_a = float(np.clip(conv_a, self.conv_a_clip_min, self.conv_a_clip_max))

        return conv_a, float(conv_a_base), float(conv_a_delta), progress, float(stagnation_boost)

    def run_once(self, actions=None):
        """执行一代二阶 CCPSO 更新。

        流程：
        1. 解析 Conv_a；
        2. 根据 pbest/gbest 构造等效引力中心 Q；
        3. 用二阶 DualC 公式得到新位置；
        4. 做隐式速度裁剪和边界裁剪；
        5. 评估适应度并更新 pbest/gbest。
        """
        conv_a, conv_a_base, conv_a_delta, progress, stagnation_boost = self._resolve_conv_a(actions)
        self.current_conv_a = conv_a
        self.current_conv_a_base = float(conv_a_base)
        self.current_conv_a_delta = float(conv_a_delta)
        self.current_conv_a_progress = float(progress)
        self.current_stagnation_boost = float(stagnation_boost)

        self.r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        self.r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))

        # 固定 Clerc 系数，当前 RL 只控制 Conv_a，不再引入额外变量。
        w = 0.729844
        c1 = 1.496180
        c2 = 1.496180
        self.current_c1 = float(c1)
        self.current_c2 = float(c2)

        c1_r1 = c1 * self.r1
        c2_r2 = c2 * self.r2
        c_gravity = c1_r1 + c2_r2

        # Q 是 pbest 和 gbest 的随机加权中心；Conv_a 不直接改变 Q。
        q = (c1_r1 * self.p_best + c2_r2 * self.history_best_x) / (c_gravity + 1e-16)
        self.current_q = q.copy()
        self.current_pbest = self.p_best.copy()
        self.current_pbest_fit = self.atom_best_fits.copy()
        self.current_pbest_diversity = self._normalized_point_diversity(self.p_best)
        self.current_gbest = self.history_best_x.copy()
        self.current_gbest_fit = float(self.history_best_fit)
        self.current_x_before_update = self.xs.copy()
        self.current_gbest_before_update = self.current_gbest.copy()

        # 二阶 DualC：同时使用 x(t) 和 x(t-1) 相对 Q 的偏移。
        a1 = 1 + w - c_gravity
        a2 = -w
        x_q = a1 * (self.xs - q) + a2 * (self.xs_old - q)
        new_xs = q + conv_a * x_q

        # 对隐式速度做与 PSO 一致的速度裁剪，保证对照实验公平。
        implicit_vs = new_xs - self.xs
        implicit_vs = np.clip(implicit_vs, self.min_v, self.max_v)

        new_xs = self.xs + implicit_vs
        new_xs = np.clip(new_xs, self.pos_min, self.pos_max)

        # 更新二阶状态：当前 x 变为下一轮的 x(t-1)。
        self.xs_old = self.xs.copy()
        self.xs = new_xs.copy()
        self.vs = implicit_vs.copy()

        self.fits = self.fun(self.xs)
        self.update_best()
        self.collect_generation_result()

        # 保存每代 Conv_a，用于 final_battle 图中的均值/方差曲线。
        self.conv_trace.append((int(self.fe_num), float(self.current_conv_a)))
