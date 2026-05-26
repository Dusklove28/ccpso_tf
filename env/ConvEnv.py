import numpy as np

from env.NormalEnv import NormalEnv


class ConvEnv(NormalEnv):
    def __init__(self, *args, reward_mode='continuous', reward_gbest_weight=8.0,
                 reward_mean_weight=2.0, reward_diversity_weight=0.5,
                 reward_instability_weight=0.3, reward_clip=2.0, **kwargs):
        super().__init__(
            *args,
            reward_mode=reward_mode,
            reward_gbest_weight=reward_gbest_weight,
            reward_mean_weight=reward_mean_weight,
            reward_diversity_weight=reward_diversity_weight,
            reward_instability_weight=reward_instability_weight,
            reward_clip=reward_clip,
            **kwargs,
        )

    def _get_progress(self):
        if self.optimizer is None:
            return 0.0
        return float(np.clip(self.optimizer.fe_num / max(self.max_fe, 1), 0.0, 1.0))

    def _get_normalized_diversity(self):
        if self.optimizer is None or not hasattr(self.optimizer, 'xs'):
            return 0.0
        diversity = np.mean(np.std(self.optimizer.xs, axis=0))
        search_span = max(float(self.optimizer.pos_max - self.optimizer.pos_min), 1e-12)
        return float(np.clip(diversity / search_span, 0.0, 1.0))

    def _get_instability_penalty(self):
        if self.optimizer is None:
            return 0.0

        boundary_ratio = 0.0
        if hasattr(self.optimizer, 'xs'):
            xs = self.optimizer.xs
            boundary_eps = 1e-12
            at_upper = xs >= self.optimizer.pos_max - boundary_eps
            at_lower = xs <= self.optimizer.pos_min + boundary_eps
            boundary_ratio = float(np.mean(np.logical_or(at_upper, at_lower)))

        velocity_clip_ratio = 0.0
        if hasattr(self.optimizer, 'vs'):
            vs = self.optimizer.vs
            max_v = max(float(abs(self.optimizer.max_v)), 1e-12)
            velocity_clip_ratio = float(np.mean(np.abs(vs) >= 0.98 * max_v))

        return float(np.clip(0.5 * boundary_ratio + 0.5 * velocity_clip_ratio, 0.0, 1.0))

    def _get_diversity_term(self):
        progress = self._get_progress()
        target_diversity = 0.15 * (1.0 - progress) + 0.02 * progress
        diversity = self._get_normalized_diversity()
        diversity_term = (diversity - target_diversity) / max(target_diversity, 1e-12)
        return float(np.clip(diversity_term, -1.0, 1.0)), float(diversity), float(target_diversity)

    def _normalized_improvement(self, old_value, new_value):
        improvement = max(0.0, float(old_value - new_value))
        scale = max(abs(float(old_value)), abs(float(new_value)), 1.0)
        return float(improvement / scale)

    def _continuous_reward(self, old_best, new_best, old_mean, new_mean):
        normalized_gbest_improvement = self._normalized_improvement(old_best, new_best)
        normalized_mean_improvement = self._normalized_improvement(old_mean, new_mean)
        gbest_reward = self.reward_gbest_weight * normalized_gbest_improvement
        mean_reward = self.reward_mean_weight * normalized_mean_improvement

        diversity_term, diversity, target_diversity = self._get_diversity_term()
        diversity_reward = self.reward_diversity_weight * diversity_term
        instability_penalty = self.reward_instability_weight * self._get_instability_penalty()

        reward = gbest_reward + mean_reward + diversity_reward - instability_penalty
        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self.last_reward_terms = {
            'reward_mode': 'continuous',
            'normalized_gbest_improvement': float(normalized_gbest_improvement),
            'normalized_mean_improvement': float(normalized_mean_improvement),
            'gbest_reward': float(gbest_reward),
            'mean_reward': float(mean_reward),
            'diversity': float(diversity),
            'target_diversity': float(target_diversity),
            'diversity_term': float(diversity_term),
            'diversity_reward': float(diversity_reward),
            'instability_penalty': float(instability_penalty),
            'reward': float(reward),
        }
        return reward

    def _calculate_reward(self, old_best, new_best, old_mean, new_mean, deta_best):
        if self.reward_mode == 'binary':
            return super()._calculate_reward(old_best, new_best, old_mean, new_mean, deta_best)
        if self.reward_mode != 'continuous':
            raise ValueError(f"unknown ConvEnv reward_mode: {self.reward_mode}")
        return self._continuous_reward(old_best, new_best, old_mean, new_mean)
