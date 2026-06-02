import copy
import os

from env.NormalEnv import NormalEnv
from matAgent.ccpso import ConvPsoSwarm
from matAgent.pso import PsoSwarm
from task.experiment_config import (
    EXPERIMENT_CCPSO_ABLATION_CONFIGS,
    EXPERIMENT_CCPSO_NOISE_SENSITIVITY_SIGMAS,
    EXPERIMENT_CCPSO_CONFIG,
    EXPERIMENT_CCPSO_ENV_CONFIG,
    EXPERIMENT_CCPSO_GAMMA,
    EXPERIMENT_CCPSO_LR_ACTOR,
    EXPERIMENT_CCPSO_LR_CRITIC,
    EXPERIMENT_DIMS,
    EXPERIMENT_CCPSO_Q_RESET_CONFIG,
    EXPERIMENT_ENV_CONFIG,
    EXPERIMENT_FUNCTIONS,
    EXPERIMENT_GROUPS,
    EXPERIMENT_GAMMA,
    EXPERIMENT_LR_ACTOR,
    EXPERIMENT_LR_CRITIC,
    EXPERIMENT_MAX_FE,
    EXPERIMENT_N_PART,
    EXPERIMENT_RUNTIMES,
    EXPERIMENT_SEPARATE_TRAINS,
    EXPERIMENT_TRAIN_MAX_EPISODE,
    EXPERIMENT_TRAIN_MAX_STEPS,
    EXPERIMENT_TRAIN_TIMES,
    EXPERIMENT_DDPG_NOISE,
    EXPERIMENT_DDPG_SIGMA,
)


CCPSO_MODE_ALIASES = {
    'a': 'CCPSO_original_reward',
    'direct_binary': 'CCPSO_original_reward',
    'direct_original': 'CCPSO_original_reward',
    'original_reward': 'CCPSO_original_reward',
    'b': 'CCPSO_continuous_reward',
    'direct_continuous': 'CCPSO_continuous_reward',
    'continuous_reward': 'CCPSO_continuous_reward',
    'b_sigma_sweep': 'CCPSO_continuous_reward_sigma_sweep',
    'direct_continuous_sigma_sweep': 'CCPSO_continuous_reward_sigma_sweep',
    'continuous_reward_sigma_sweep': 'CCPSO_continuous_reward_sigma_sweep',
    'c': 'CCPSO_progress_prior',
    'progress_binary': 'CCPSO_progress_prior',
    'progress_original': 'CCPSO_progress_prior',
    'progress_prior_binary': 'CCPSO_progress_prior',
    'd': 'CCPSO_DualC_full',
    'full': 'CCPSO_DualC_full',
    'progress_continuous': 'CCPSO_DualC_full',
    'progress_prior_continuous': 'CCPSO_DualC_full',
}


CCPSO_Q_RESET_MODE_ALIASES = {
    'a_q_reset': 'CCPSO_original_reward',
    'direct_binary_q_reset': 'CCPSO_original_reward',
    'direct_original_q_reset': 'CCPSO_original_reward',
    'original_reward_q_reset': 'CCPSO_original_reward',
    'b_q_reset': 'CCPSO_continuous_reward',
    'direct_continuous_q_reset': 'CCPSO_continuous_reward',
    'continuous_reward_q_reset': 'CCPSO_continuous_reward',
    'c_q_reset': 'CCPSO_progress_prior',
    'progress_binary_q_reset': 'CCPSO_progress_prior',
    'progress_original_q_reset': 'CCPSO_progress_prior',
    'progress_prior_binary_q_reset': 'CCPSO_progress_prior',
    'q_reset': 'CCPSO_progress_prior',
    'd_q_reset': 'CCPSO_DualC_full',
    'full_q_reset': 'CCPSO_DualC_full',
    'progress_continuous_q_reset': 'CCPSO_DualC_full',
    'progress_prior_continuous_q_reset': 'CCPSO_DualC_full',
}


TRUE_ENV_VALUES = {'1', 'true', 'yes', 'y', 'on'}


def _format_sigma_label(sigma):
    return str(sigma).replace('.', 'p')


def _read_float_env(name):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got: {value}") from exc


def _read_bool_env(name):
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in TRUE_ENV_VALUES


def _apply_q_reset_config(pair, rename=True):
    pair = copy.deepcopy(pair)
    optimizer_config = copy.deepcopy(pair.get('optimizer_config', {}))
    optimizer_config.update(EXPERIMENT_CCPSO_Q_RESET_CONFIG)
    pair['optimizer_config'] = optimizer_config
    if rename and not pair['name'].endswith('_q_reset'):
        old_name = pair['name']
        old_phase_name = pair.get('phase_name', old_name)
        pair['name'] = f"{old_name}_q_reset"
        pair['phase_name'] = f"{old_phase_name}_q_reset"
    return pair


def _apply_noise_config(pair, sigma=None, noise=None, rename=True):
    pair = copy.deepcopy(pair)
    pair['noise'] = noise or os.environ.get('CCPSO_NOISE') or os.environ.get('DDPG_NOISE') or EXPERIMENT_DDPG_NOISE
    pair['sigma'] = EXPERIMENT_DDPG_SIGMA if sigma is None else float(sigma)
    if rename:
        old_name = pair['name']
        old_phase_name = pair.get('phase_name', old_name)
        suffix = f"_sigma_{_format_sigma_label(pair['sigma'])}"
        pair['name'] = f"{old_name}{suffix}"
        pair['phase_name'] = f"{old_phase_name}{suffix}"
    return pair


def _apply_env_noise_override(pair):
    sigma = _read_float_env('CCPSO_SIGMA')
    if sigma is None:
        sigma = _read_float_env('DDPG_SIGMA')
    if sigma is None and not (os.environ.get('CCPSO_NOISE') or os.environ.get('DDPG_NOISE')):
        return pair
    return _apply_noise_config(pair, sigma=sigma, rename=True)


def _apply_env_q_reset_override(pair):
    if not (
        _read_bool_env('CCPSO_Q_RESET')
        or _read_bool_env('CCPSO_ANTI_Q_COLLAPSE')
        or _read_bool_env('ANTI_Q_COLLAPSE')
    ):
        return pair
    return _apply_q_reset_config(pair, rename=True)


def _env_class_from_name(name):
    env_classes = {
        'NormalEnv': NormalEnv,
    }
    if name not in env_classes:
        raise ValueError(f"unknown env_class in experiment config: {name}")
    return env_classes[name]


def ccpso_ablation_optimizer_pairs():
    optimizer_pairs = []
    for config in EXPERIMENT_CCPSO_ABLATION_CONFIGS:
        optimizer_pairs.append({
            'name': config['name'],
            'phase_name': config['name'],
            'train_optimizer': ConvPsoSwarm,
            'evaluate_optimizer': ConvPsoSwarm,
            'env_class': _env_class_from_name(config['env_class']),
            'optimizer_config': copy.deepcopy(config['optimizer_config']),
            'env_config': copy.deepcopy(config['env_config']),
            'lr_actor': config['lr_actor'],
            'lr_critic': config['lr_critic'],
            'gamma': config['gamma'],
        })
    return optimizer_pairs


def _default_ccpso_optimizer_pair():
    return {
        'name': 'CCPSO_DualC_reward_mode',
        'train_optimizer': ConvPsoSwarm,
        'evaluate_optimizer': ConvPsoSwarm,
        'env_class': NormalEnv,
        'optimizer_config': EXPERIMENT_CCPSO_CONFIG,
        'env_config': EXPERIMENT_CCPSO_ENV_CONFIG,
        'lr_actor': EXPERIMENT_CCPSO_LR_ACTOR,
        'lr_critic': EXPERIMENT_CCPSO_LR_CRITIC,
        'gamma': EXPERIMENT_CCPSO_GAMMA,
    }


def _ccpso_continuous_reward_sigma_sweep_pairs():
    base_pair = None
    for pair in ccpso_ablation_optimizer_pairs():
        if pair['name'].startswith('CCPSO_continuous_reward'):
            base_pair = pair
            break
    if base_pair is None:
        raise ValueError("CCPSO_continuous_reward config is missing.")
    return [
        _apply_env_q_reset_override(_apply_noise_config(base_pair, sigma=sigma, rename=True))
        for sigma in EXPERIMENT_CCPSO_NOISE_SENSITIVITY_SIGMAS
    ]


def _selected_ccpso_optimizer_pairs():
    mode_spec = os.environ.get('CCPSO_MODE') or os.environ.get('CCPSO_EXPERIMENT_MODE')
    if not mode_spec:
        return [_apply_env_q_reset_override(_apply_env_noise_override(_default_ccpso_optimizer_pair()))]

    mode_spec = mode_spec.strip()
    if mode_spec.lower() in ('b_sigma_sweep', 'direct_continuous_sigma_sweep', 'continuous_reward_sigma_sweep'):
        return _ccpso_continuous_reward_sigma_sweep_pairs()

    all_pairs = ccpso_ablation_optimizer_pairs()
    if mode_spec.lower() in ('all', 'ablation', 'all_ablation'):
        return [_apply_env_q_reset_override(_apply_env_noise_override(pair)) for pair in all_pairs]
    if mode_spec.lower() in ('all_q_reset', 'ablation_q_reset', 'all_ablation_q_reset'):
        return [_apply_env_noise_override(_apply_q_reset_config(pair)) for pair in all_pairs]

    pair_by_name = {pair['name']: pair for pair in all_pairs}
    selected_pairs = []
    for raw_mode in mode_spec.split(','):
        mode = raw_mode.strip().lower()
        if not mode:
            continue
        q_reset_requested = mode in CCPSO_Q_RESET_MODE_ALIASES
        pair_name = CCPSO_Q_RESET_MODE_ALIASES.get(mode) or CCPSO_MODE_ALIASES.get(mode, raw_mode.strip())
        if pair_name not in pair_by_name:
            supported = ', '.join(sorted(set(CCPSO_MODE_ALIASES) | set(CCPSO_Q_RESET_MODE_ALIASES)))
            raise ValueError(f"unknown CCPSO_MODE: {raw_mode}. supported aliases: {supported}, all")
        pair = copy.deepcopy(pair_by_name[pair_name])
        if q_reset_requested:
            pair = _apply_q_reset_config(pair)
        pair = _apply_env_q_reset_override(_apply_env_noise_override(pair))
        selected_pairs.append(pair)

    if not selected_pairs:
        raise ValueError("CCPSO_MODE was set but no valid mode was selected.")
    return selected_pairs


def all_tasks_generate():
    rl_optimizer_pairs = [
        {
            'name': 'RLPSO_original_env',
            'train_optimizer': PsoSwarm,
            'evaluate_optimizer': PsoSwarm,
            'env_class': NormalEnv,
            'optimizer_config': {},
            'env_config': EXPERIMENT_ENV_CONFIG,
            'lr_actor': EXPERIMENT_LR_ACTOR,
            'lr_critic': EXPERIMENT_LR_CRITIC,
            'gamma': EXPERIMENT_GAMMA,
        },
    ]
    rl_optimizer_pairs.extend(_selected_ccpso_optimizer_pairs())

    task = {
        'type': 'top',
        'baseline_optimizers': [PsoSwarm],
        'rl_optimizer_pairs': rl_optimizer_pairs,
        'evaluate_function': EXPERIMENT_FUNCTIONS,
        'runtimes': EXPERIMENT_RUNTIMES,
        'separate_trains': EXPERIMENT_SEPARATE_TRAINS,
        'groups': EXPERIMENT_GROUPS,
        'train_max_episode': EXPERIMENT_TRAIN_MAX_EPISODE,
        'train_max_steps': EXPERIMENT_TRAIN_MAX_STEPS,
        'dims': EXPERIMENT_DIMS,
        'train_times': EXPERIMENT_TRAIN_TIMES,
        'max_fe': EXPERIMENT_MAX_FE,
        'n_part': EXPERIMENT_N_PART,
        'lr_critic': EXPERIMENT_LR_CRITIC,
        'lr_actor': EXPERIMENT_LR_ACTOR,
        'gamma': EXPERIMENT_GAMMA,
    }

    return [task]


if __name__ == '__main__':
    print(all_tasks_generate())
