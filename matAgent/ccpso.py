import numpy as np
from matAgent.baseAgent import MatSwarm



class ConvPsoSwarm(MatSwarm):
    optimizer_name = 'Conv_PSO_DualC'
    action_space = 1
    obs_space = 15
    CONV_A_SCHEDULES = {'direct', 'progress_prior'}

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = self.optimizer_name
        self.fixed_conv_a = self.config.get('fixed_conv_a')
        if self.fixed_conv_a is not None:
            self.fixed_conv_a = float(self.fixed_conv_a)
        self.ccpso_update_mode = self.config.get('ccpso_update_mode', 'second_order')
        self.conv_a_schedule = self.config.get('conv_a_schedule', 'progress_prior')
        if self.conv_a_schedule not in self.CONV_A_SCHEDULES:
            schedules = ', '.join(sorted(self.CONV_A_SCHEDULES))
            raise ValueError(f"unknown conv_a_schedule: {self.conv_a_schedule}. supported: {schedules}")
        self.conv_a_max = float(self.config.get('conv_a_max', 1.5))
        self.conv_a_min = float(self.config.get('conv_a_min', 0.2))
        self.conv_a_delta_scale = float(self.config.get('conv_a_delta_scale', 0.2))
        self.conv_a_clip_min = float(self.config.get('conv_a_clip_min', 0.05))
        self.conv_a_clip_max = float(self.config.get('conv_a_clip_max', 1.8))
        self.stagnation_boost_max = float(self.config.get('stagnation_boost_max', 0.25))
        self.stagnation_boost_fe_ratio = float(self.config.get('stagnation_boost_fe_ratio', 0.2))
        self.first_order_sigma_floor = float(self.config.get('first_order_sigma_floor', 0.001))
        self.anti_q_collapse = bool(self.config.get('anti_q_collapse', False))
        self.collapse_min_progress = float(self.config.get('collapse_min_progress', 0.4))
        self.collapse_stagnation_fe_ratio = float(self.config.get('collapse_stagnation_fe_ratio', 0.12))
        self.collapse_gbest_improvement_threshold = float(
            self.config.get('collapse_gbest_improvement_threshold', 1e-3)
        )
        self.collapse_pbest_diversity_threshold = float(
            self.config.get('collapse_pbest_diversity_threshold', 0.005)
        )
        self.collapse_reset_ratio = float(self.config.get('collapse_reset_ratio', 0.2))
        self.collapse_reset_cooldown_fe_ratio = float(self.config.get('collapse_reset_cooldown_fe_ratio', 0.1))
        self.collapse_restart_radius_ratio = float(self.config.get('collapse_restart_radius_ratio', 0.08))
        self.collapse_global_restart_probability = float(
            self.config.get('collapse_global_restart_probability', 0.25)
        )
        self.last_pbest_reset_fe = -np.inf
        self.pending_q_reset_indices = None
        # 追踪收敛系数Conv_a
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
        self.current_gbest_window_relative_improvement = None
        self.current_pbest_reset_count = 0
        self.gbest_window_trace = []
        self.pbest_reset_trace = []
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
        self.gbest_window_trace = [(int(self.fe_num), float(self.history_best_fit))]
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

    def _normalized_point_diversity(self, points):
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return 0.0
        search_span = max(float(self.pos_max - self.pos_min), 1e-12)
        return float(np.mean(np.std(points, axis=0)) / search_span)

    def _record_gbest_window_value(self):
        fe = int(self.fe_num)
        item = (fe, float(self.history_best_fit))
        if self.gbest_window_trace and self.gbest_window_trace[-1][0] == fe:
            self.gbest_window_trace[-1] = item
        else:
            self.gbest_window_trace.append(item)

        window_fe = max(int(round(self.fe_max * self.collapse_stagnation_fe_ratio)), self.n_part)
        keep_after_fe = fe - max(window_fe * 2, self.n_part * 2)
        while len(self.gbest_window_trace) > 1 and self.gbest_window_trace[1][0] <= keep_after_fe:
            self.gbest_window_trace.pop(0)

    def _get_gbest_window_relative_improvement(self):
        window_fe = max(int(round(self.fe_max * self.collapse_stagnation_fe_ratio)), self.n_part)
        target_fe = int(self.fe_num - window_fe)
        if target_fe < 0 or not self.gbest_window_trace:
            return None, window_fe

        anchor_fit = None
        for fe, fit in reversed(self.gbest_window_trace):
            if fe <= target_fe:
                anchor_fit = float(fit)
                break
        if anchor_fit is None:
            return None, window_fe

        current_fit = float(self.history_best_fit)
        improvement = max(anchor_fit - current_fit, 0.0)
        scale = max(abs(anchor_fit), abs(current_fit), 1.0)
        return float(improvement / scale), window_fe

    def _apply_pending_q_reset(self):
        if self.pending_q_reset_indices is None:
            return 0

        selected = np.asarray(self.pending_q_reset_indices, dtype=int)
        self.pending_q_reset_indices = None
        if selected.size == 0:
            return 0

        span = float(self.pos_max - self.pos_min)
        progress = self._get_progress()
        radius = span * self.collapse_restart_radius_ratio * max(1.0 - progress, 0.25)
        center = self.history_best_x.reshape(1, -1)

        new_positions = center + np.random.normal(0.0, radius, (selected.size, self.n_dim))
        global_mask = np.random.uniform(0.0, 1.0, selected.size) < self.collapse_global_restart_probability
        if np.any(global_mask):
            new_positions[global_mask] = np.random.uniform(
                self.pos_min,
                self.pos_max,
                (int(np.sum(global_mask)), self.n_dim),
            )
        new_positions = np.clip(new_positions, self.pos_min, self.pos_max)

        new_vs = np.random.uniform(self.min_v, self.max_v, (selected.size, self.n_dim)) * 0.1
        self.xs[selected] = new_positions
        self.vs[selected] = new_vs
        self.xs_old[selected] = new_positions - new_vs
        self.p_best[selected] = new_positions.copy()
        self.atom_best_fits[selected] = np.inf
        return int(selected.size)

    def _maybe_reset_collapsed_pbest(self):
        if not self.anti_q_collapse:
            self.current_pbest_reset_count = 0
            return 0
        if self.pending_q_reset_indices is not None:
            self.current_pbest_reset_count = 0
            return 0

        progress = self._get_progress()
        if progress < self.collapse_min_progress:
            self.current_pbest_reset_count = 0
            return 0

        window_relative_improvement, window_fe = self._get_gbest_window_relative_improvement()
        self.current_gbest_window_relative_improvement = window_relative_improvement
        if window_relative_improvement is None:
            self.current_pbest_reset_count = 0
            return 0
        if window_relative_improvement >= self.collapse_gbest_improvement_threshold:
            self.current_pbest_reset_count = 0
            return 0

        cooldown_fe = max(self.fe_max * self.collapse_reset_cooldown_fe_ratio, self.n_part)
        if self.fe_num - self.last_pbest_reset_fe < cooldown_fe:
            self.current_pbest_reset_count = 0
            return 0

        pbest_diversity = self._normalized_point_diversity(self.p_best)
        self.current_pbest_diversity = pbest_diversity
        if pbest_diversity > self.collapse_pbest_diversity_threshold:
            self.current_pbest_reset_count = 0
            return 0

        reset_count = max(1, int(round(self.n_part * self.collapse_reset_ratio)))
        reset_count = min(reset_count, self.n_part - 1)
        if reset_count <= 0:
            self.current_pbest_reset_count = 0
            return 0

        best_index = int(np.argmin(self.atom_best_fits))
        candidate_indices = [idx for idx in np.argsort(self.atom_best_fits)[::-1] if idx != best_index]
        selected = np.asarray(candidate_indices[:reset_count], dtype=int)
        if selected.size == 0:
            self.current_pbest_reset_count = 0
            return 0

        self.pending_q_reset_indices = selected.copy()
        self.last_pbest_reset_fe = self.fe_num
        self.current_pbest_reset_count = int(selected.size)
        self.pbest_reset_trace.append((
            int(self.fe_num),
            int(selected.size),
            float(pbest_diversity),
            int(window_fe),
            float(window_relative_improvement),
        ))
        return int(selected.size)

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
        elif self.conv_a_schedule == 'direct':
            # 兼容旧实验：Actor 直接输出 Conv_a - 1。
            conv_a_base = 1.0
            conv_a_delta = raw_action
            stagnation_boost = 0.0
            conv_a = raw_action + 1.0
        else:
            raise ValueError(f"unknown conv_a_schedule: {self.conv_a_schedule}")

        conv_a = float(np.clip(conv_a, self.conv_a_clip_min, self.conv_a_clip_max))
        return conv_a, float(conv_a_base), float(conv_a_delta), progress, float(stagnation_boost)

    def run_once(self, actions=None):
        applied_reset_count = self._apply_pending_q_reset()
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
        self.current_q = Q.copy()
        self.current_pbest = self.p_best.copy()
        self.current_pbest_fit = self.atom_best_fits.copy()
        self.current_pbest_diversity = self._normalized_point_diversity(self.p_best)
        self.current_gbest = self.history_best_x.copy()
        self.current_gbest_fit = float(self.history_best_fit)
        self.current_x_before_update = self.xs.copy()
        self.current_gbest_before_update = self.current_gbest.copy()

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
        self._record_gbest_window_value()
        scheduled_reset_count = self._maybe_reset_collapsed_pbest()
        self.current_pbest_reset_count = max(applied_reset_count, scheduled_reset_count)
        self.conv_trace.append((int(self.fe_num), float(self.current_conv_a)))
