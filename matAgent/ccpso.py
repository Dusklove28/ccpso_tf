import numpy as np

from env.CcPSOEnv import (
    CCPSO_STATE_DIM,
    build_ccpso_state,
    compute_ccpso_collapse_risk,
    normalized_improvement,
)
from matAgent.baseAgent import MatSwarm


class ConvPsoSwarm(MatSwarm):
    """Second-order CCPSO controlled by DDPG.

    The Q-centered DualC mechanism is fixed. The actor directly controls the
    convergence coefficient Conv_a; no hand-written progress prior is added in
    this research line.
    """

    optimizer_name = 'Conv_PSO_DualC'
    action_space = 1
    obs_space = CCPSO_STATE_DIM

    def __init__(self, n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic):
        super().__init__(n_run, n_part, show, fun, n_dim, pos_max, pos_min, config_dic)
        self.name = self.optimizer_name

        self.fixed_conv_a = self.config.get('fixed_conv_a')
        if self.fixed_conv_a is not None:
            self.fixed_conv_a = float(self.fixed_conv_a)

        self.conv_a_clip_min = float(self.config.get('conv_a_clip_min', 0.00))
        self.conv_a_clip_max = float(self.config.get('conv_a_clip_max', 2.0))

        # Fixed CCPSO coefficients. They are not actor-controlled in V1.
        self.ccpso_w = float(self.config.get('ccpso_w', 0.729844))
        self.ccpso_c1 = float(self.config.get('ccpso_c1', 1.496180))
        self.ccpso_c2 = float(self.config.get('ccpso_c2', 1.496180))

        self.anti_collapse_fe_ratio = float(
            self.config.get('anti_collapse_fe_ratio', self.config.get('stagnation_boost_fe_ratio', 0.2))
        )
        self.anti_collapse_q_div_threshold = float(
            self.config.get('anti_collapse_q_div_threshold', 0.03)
        )
        self.anti_collapse_q_gbest_threshold = float(
            self.config.get('anti_collapse_q_gbest_threshold', 0.05)
        )

        # Diagnosis fields used by plots and evaluate/q_collapse_diagnosis.py.
        self.current_conv_a = None
        self.current_conv_a_base = None
        self.current_conv_a_delta = None
        self.current_raw_action = 0.0
        self.current_conv_a_norm = 0.0
        self.current_conv_a_progress = None
        self.current_stagnation_boost = 0.0
        self.current_anti_collapse_boost = 0.0
        self.current_collapse_risk = 0.0
        self.current_guard_strength = 0.0
        self.current_collapse_metrics = {}
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

        self.recent_gbest_improvement = 0.0
        self.recent_mean_improvement = 0.0

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
        self.last_best_update_fe = self.fe_num
        self.run_flag = self.fe_num < self.fe_max
        if (self.fe_num % self.record_per_fe == 0 or self.fe_num == self.fe_max) and self.fe_num <= self.fe_max:
            self.data_collect_method()

        # DualC needs x(t-1). Initialization estimates it from x(t)-v(t).
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

    def get_state(self):
        return build_ccpso_state(self)

    def _get_progress(self):
        return float(np.clip(self.fe_num / max(self.fe_max, 1), 0.0, 1.0))

    def _normalized_point_diversity(self, points):
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return 0.0
        search_span = max(float(self.pos_max - self.pos_min), 1e-12)
        return float(np.mean(np.std(points, axis=0)) / search_span)

    def _normalized_mean_row_distance(self, left, right):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        if left.size == 0 or right.size == 0:
            return 0.0
        if right.ndim == 1:
            right = np.broadcast_to(right, left.shape)
        search_span = max(float(self.pos_max - self.pos_min), 1e-12)
        norm = max(np.sqrt(float(self.n_dim)) * search_span, 1e-12)
        return float(np.mean(np.linalg.norm(left - right, axis=1)) / norm)

    def _normalize_actions(self, actions):
        if actions is None:
            actions = np.zeros(self.action_space, dtype=float)
        elif hasattr(actions, 'numpy'):
            actions = actions.numpy()
        actions = np.asarray(actions, dtype=float).reshape(-1)

        raw_action = float(actions[0]) if actions.size > 0 else 0.0
        return float(np.clip(raw_action, -1.0, 1.0))

    def _resolve_conv_a(self, actions):
        progress = self._get_progress()
        if self.fixed_conv_a is not None:
            conv_a = float(np.clip(self.fixed_conv_a, self.conv_a_clip_min, self.conv_a_clip_max))
            conv_a_norm = (conv_a - self.conv_a_clip_min) / max(self.conv_a_clip_max - self.conv_a_clip_min, 1e-12)
            collapse_risk, collapse_metrics = compute_ccpso_collapse_risk(self)
            return conv_a, progress, 0.0, float(conv_a_norm), float(collapse_risk), collapse_metrics

        raw_action = self._normalize_actions(actions)
        conv_a_norm = 0.5 * (raw_action + 1.0)
        conv_a = self.conv_a_clip_min + conv_a_norm * (self.conv_a_clip_max - self.conv_a_clip_min)
        conv_a = float(np.clip(conv_a, self.conv_a_clip_min, self.conv_a_clip_max))
        collapse_risk, collapse_metrics = compute_ccpso_collapse_risk(self)
        return conv_a, progress, float(raw_action), float(conv_a_norm), float(collapse_risk), collapse_metrics

    def _current_instability_components(self):
        boundary_eps = 1e-12
        at_upper = self.xs >= self.pos_max - boundary_eps
        at_lower = self.xs <= self.pos_min + boundary_eps
        boundary_ratio = float(np.mean(np.logical_or(at_upper, at_lower)))

        max_v = max(float(abs(self.max_v)), 1e-12)
        velocity_clip_ratio = float(np.mean(np.abs(self.vs) >= 0.98 * max_v))
        instability_penalty = float(np.clip(0.5 * boundary_ratio + 0.5 * velocity_clip_ratio, 0.0, 1.0))
        return boundary_ratio, velocity_clip_ratio, instability_penalty

    def run_once(self, actions=None):
        old_best = float(self.history_best_fit)
        old_mean = float(np.mean(self.fits))

        conv_a, progress, raw_action, conv_a_norm, collapse_risk, collapse_metrics = self._resolve_conv_a(actions)
        self.current_conv_a = conv_a
        self.current_conv_a_base = 0.0
        self.current_conv_a_delta = 0.0
        self.current_raw_action = float(raw_action)
        self.current_conv_a_norm = float(conv_a_norm)
        self.current_conv_a_progress = float(progress)
        self.current_stagnation_boost = 0.0
        self.current_anti_collapse_boost = 0.0
        self.current_collapse_risk = float(collapse_risk)
        self.current_guard_strength = 0.0
        self.current_collapse_metrics = dict(collapse_metrics)

        self.r1 = np.random.uniform(0, 1, (self.n_part, self.n_dim))
        self.r2 = np.random.uniform(0, 1, (self.n_part, self.n_dim))

        w = self.ccpso_w
        c1 = self.ccpso_c1
        c2 = self.ccpso_c2
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

        new_mean = float(np.mean(self.fits))
        old_history_best = old_best
        self.recent_gbest_improvement = normalized_improvement(
            old_best,
            self.history_best_fit,
            positive_only=True,
        )
        self.recent_mean_improvement = normalized_improvement(
            old_mean,
            new_mean,
            positive_only=False,
        )

        self.collect_generation_result()
        q_diversity = self._normalized_point_diversity(self.current_q)
        q_gbest_distance = self._normalized_mean_row_distance(self.current_q, self.history_best_x)
        x_q_distance = self._normalized_mean_row_distance(self.xs, self.current_q)
        swarm_diversity = self._normalized_point_diversity(self.xs)
        pbest_diversity = self._normalized_point_diversity(self.p_best)
        boundary_ratio, velocity_clip_ratio, instability_penalty = self._current_instability_components()

        self.conv_trace.append({
            "fe": int(self.fe_num),
            "conv_a": float(self.current_conv_a),
            "raw_action": float(self.current_raw_action),
            "conv_a_norm": float(self.current_conv_a_norm),
            "progress": float(self.current_conv_a_progress),
            "swarm_diversity": float(swarm_diversity),
            "pbest_diversity": float(pbest_diversity),
            "q_diversity": float(q_diversity),
            "q_gbest_distance": float(q_gbest_distance),
            "x_q_distance": float(x_q_distance),
            "collapse_risk": float(self.current_collapse_risk),
            "recent_gbest_improvement": float(self.recent_gbest_improvement),
            "recent_mean_improvement": float(self.recent_mean_improvement),
            "boundary_ratio": float(boundary_ratio),
            "velocity_clip_ratio": float(velocity_clip_ratio),
            "instability_penalty": float(instability_penalty),
            "gbest_fit_before": float(old_history_best),
            "gbest_fit_after": float(self.history_best_fit),
            "mean_fit_before": float(old_mean),
            "mean_fit_after": float(new_mean),
        })
