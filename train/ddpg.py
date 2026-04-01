from log import logger
from rl.DDPG.TF2_DDPG_Basic import DDPG


DEFAULT_ACTOR_UNITS = (16, 32, 32, 32, 64, 64)
DEFAULT_CRITIC_UNITS = (8, 16, 32, 32, 16, 8)


def get_ddpg_object(
        env,
        discrete=False,
        use_priority=True,
        lr_actor=1e-5,
        lr_critic=1e-3,
        actor_units=None,
        critic_units=None,
        noise='norm',
        sigma=0.15,
        tau=0.125,
        gamma=0.85,
        batch_size=64,
        memory_cap=100000):
    if actor_units is None:
        actor_units = DEFAULT_ACTOR_UNITS
    if critic_units is None:
        critic_units = DEFAULT_CRITIC_UNITS

    return DDPG(
        env,
        discrete=discrete,
        use_priority=use_priority,
        lr_critic=lr_critic,
        lr_actor=lr_actor,
        actor_units=actor_units,
        critic_units=critic_units,
        noise=noise,
        sigma=sigma,
        tau=tau,
        gamma=gamma,
        batch_size=batch_size,
        memory_cap=memory_cap,
    )


def train():
    from env.NormalEnv import NormalEnv
    from matAgent.pso import PsoSwarm
    from task.experiment_config import (
        EXPERIMENT_FUNCTIONS,
        EXPERIMENT_LR_ACTOR,
        EXPERIMENT_LR_CRITIC,
        EXPERIMENT_MAX_FE,
        EXPERIMENT_N_PART,
        EXPERIMENT_TRAIN_MAX_EPISODE,
        EXPERIMENT_TRAIN_MAX_STEPS,
        get_primary_experiment_signature,
    )

    signature = get_primary_experiment_signature()
    gym_env = NormalEnv(
        obs_shape=(PsoSwarm.obs_space,),
        action_shape=(PsoSwarm.action_space * signature['group'],),
        target_optimizer=PsoSwarm,
        fun_nums=[EXPERIMENT_FUNCTIONS[0]],
        max_fe=EXPERIMENT_MAX_FE,
        n_part=EXPERIMENT_N_PART,
        n_dim=signature['dim'],
        group=signature['group'],
    )

    logger.info('running local DDPG smoke-train on the main PSO environment')
    ddpg = get_ddpg_object(
        gym_env,
        discrete=False,
        memory_cap=10000000,
        lr_actor=EXPERIMENT_LR_ACTOR,
        lr_critic=EXPERIMENT_LR_CRITIC,
    )
    ddpg.train(
        max_episodes=min(EXPERIMENT_TRAIN_MAX_EPISODE, 5),
        max_epochs=min(EXPERIMENT_TRAIN_MAX_STEPS, 500),
        max_steps=min(EXPERIMENT_TRAIN_MAX_STEPS, 500),
        save_freq=1,
    )


def test():
    pass


if __name__ == "__main__":
    logger.info('running ddpg.py directly')
    train()
