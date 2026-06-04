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

        self.conv_a_delta_scale = float(self.config.get('conv_a_delta_scale', 0.2))
        self.conv_a_clip_min = float(self.config.get('conv_a_clip_min', 0.05))
        self.conv_a_clip_max = float(self.config.get('conv_a_clip_max', 1.8))
        self.stagnation_boost_max = float(self.config.get('stagnation_boost_max', 0.25))
        self.stagnation_boost_fe_ratio = float(self.config.get('stagnation_boost_fe_ratio', 0.2))

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

        self.vs = np.zeros_like(self.xs)
        self.p_best = np.zeros_like(self.xs)
        self.atom_best_fits = np.zeros(self.n_part)
        self.g_best = np.zeros(n_dim)
        self.fits = np.zeros(self.n_part)

        self.r1 = np.zeros((self.n_part, self.n_dim))
        self.r2 = np.zeros((self.n_part, self.n_dim))

        self.init()

    def init(self):
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

        self.xs_old = self.xs - self.vs

    def set_x(self, x):
        assert x.shape == self.xs.shape
        self.xs = x

    def update_best(self):
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

    def _progress_prior_base(self, progress):
        if progress <= 0.6:
            return 1.5 - 0.48 * (progress / 0.6) ** 1.2
        return 1.0 - 0.8 * ((progress - 0.6) / 0.4) ** 0.7

    def _resolve_conv_a(self, actions):
        progress = self._get_progress()
        if self.fixed_conv_a is not None:
            conv_a = float(np.clip(self.fixed_conv_a, self.conv_a_clip_min, self.conv_a_clip_max))
            return conv_a, conv_a, 0.0, progress, 0.0

        if actions is None:
            actions = np.zeros(self.action_space, dtype=float)
        elif hasattr(actions, 'numpy'):
            actions = actions.numpy()

        actions = np.asarray(actions, dtype=float).reshape(-1)
        raw_action = float(actions[0]) if actions.size else 0.0

        conv_a_base = self._progress_prior_base(progress)
        conv_a_delta = raw_action * self.conv_a_delta_scale
        stagnation_boost = self._get_stagnation_boost()
        conv_a = conv_a_base + conv_a_delta + stagnation_boost
        conv_a = float(np.clip(conv_a, self.conv_a_clip_min, self.conv_a_clip_max))

        return conv_a, float(conv_a_base), float(conv_a_delta), progress, float(stagnation_boost)

    def run_once(self, actions=None):
        conv_a, conv_a_base, conv_a_delta, progress, stagnation_boost = self._resolve_conv_a(actions)
        self.current_conv_a = conv_a
        self.current_conv_a_base = float(conv_a_base)
        self.current_conv_a_delta = float(conv_a_delta)
        self.current_conv_a_progress = float(progress)
        self.current_stagnation_boost = float(stagnation_boost)

        self.r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        self.r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))

        w = 0.729844
        c1 = 1.496180
        c2 = 1.496180
        self.current_c1 = float(c1)
        self.current_c2 = float(c2)

        c1_r1 = c1 * self.r1
        c2_r2 = c2 * self.r2
        c_gravity = c1_r1 + c2_r2

        q = (c1_r1 * self.p_best + c2_r2 * self.history_best_x) / (c_gravity + 1e-16)
        self.current_q = q.copy()
        self.current_pbest = self.p_best.copy()
        self.current_pbest_fit = self.atom_best_fits.copy()
        self.current_pbest_diversity = self._normalized_point_diversity(self.p_best)
        self.current_gbest = self.history_best_x.copy()
        self.current_gbest_fit = float(self.history_best_fit)
        self.current_x_before_update = self.xs.copy()
        self.current_gbest_before_update = self.current_gbest.copy()

        a1 = 1 + w - c_gravity
        a2 = -w
        x_q = a1 * (self.xs - q) + a2 * (self.xs_old - q)
        new_xs = q + conv_a * x_q

        implicit_vs = new_xs - self.xs
        implicit_vs = np.clip(implicit_vs, self.min_v, self.max_v)

        new_xs = self.xs + implicit_vs
        new_xs = np.clip(new_xs, self.pos_min, self.pos_max)

        self.xs_old = self.xs.copy()
        self.xs = new_xs.copy()
        self.vs = implicit_vs.copy()

        self.fits = self.fun(self.xs)
        self.update_best()
        self.collect_generation_result()
        self.conv_trace.append((int(self.fe_num), float(self.current_conv_a)))
