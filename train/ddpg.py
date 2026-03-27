from log import logger
from rl.DDPG.TF2_DDPG_Basic import DDPG


ORIGINAL_RLEPSO_ACTOR_UNITS = (16, 32, 32, 32, 64, 64)
ORIGINAL_RLEPSO_CRITIC_UNITS = (8, 16, 32, 32, 16, 8)
ORIGINAL_RLEPSO_LR_CRITIC = 1e-7
ORIGINAL_RLEPSO_LR_ACTOR = 1e-9
ORIGINAL_RLEPSO_MAX_EPISODES = 200
ORIGINAL_RLEPSO_MAX_EPOCHS = 8000
ORIGINAL_RLEPSO_MAX_STEPS = 1000
ORIGINAL_RLEPSO_SAVE_FREQ = 50
ORIGINAL_RLEPSO_MEMORY_CAP = 10000000


def build_original_rlepso_ddpg(
        env,
        discrete=False,
        noise='norm',
        sigma=0.15,
        tau=0.125,
        gamma=0.85,
        batch_size=64,
        memory_cap=100000):
    return DDPG(
        env,
        discrete=discrete,
        memory_cap=memory_cap,
        actor_units=ORIGINAL_RLEPSO_ACTOR_UNITS,
        critic_units=ORIGINAL_RLEPSO_CRITIC_UNITS,
        use_priority=True,
        lr_critic=ORIGINAL_RLEPSO_LR_CRITIC,
        lr_actor=ORIGINAL_RLEPSO_LR_ACTOR,
        noise=noise,
        sigma=sigma,
        tau=tau,
        gamma=gamma,
        batch_size=batch_size,
    )


def get_original_rlepso_train_config():
    return {
        'max_episodes': ORIGINAL_RLEPSO_MAX_EPISODES,
        'max_epochs': ORIGINAL_RLEPSO_MAX_EPOCHS,
        'max_steps': ORIGINAL_RLEPSO_MAX_STEPS,
        'save_freq': ORIGINAL_RLEPSO_SAVE_FREQ,
        'memory_cap': ORIGINAL_RLEPSO_MEMORY_CAP,
    }


def build_original_rlepso_train_env(al_type='testpso', fun_nums=None, show=False):
    from env.TestpsoEnv import TestpsoEnv

    fixed_fun_num = None
    if fun_nums and len(fun_nums) == 1:
        fixed_fun_num = fun_nums[0]

    return TestpsoEnv(show=show, al_type=al_type, fixed_fun_num=fixed_fun_num)


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
    # Keep RL_testpso on its original training settings.
    return build_original_rlepso_ddpg(
        env,
        discrete=discrete,
        noise=noise,
        sigma=sigma,
        tau=tau,
        gamma=gamma,
        batch_size=batch_size,
        memory_cap=memory_cap,
    )


def train():
    """
    Local debug entry only. The task framework does not call this directly.
    """
    algo_type = 'testpso'
    logger.info(f"==== local debug training, algorithm: {algo_type} ====")

    gym_env = build_original_rlepso_train_env(al_type=algo_type, show=False)

    try:
        assert gym_env.action_space.high == -gym_env.action_space.low
        is_discrete = False
        logger.info('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        logger.info('Discrete Action Space')

    ddpg = get_ddpg_object(
        gym_env,
        discrete=is_discrete,
        memory_cap=ORIGINAL_RLEPSO_MEMORY_CAP,
        noise='norm',
        sigma=0.15,
        tau=0.125,
        gamma=0.85,
    )

    logger.info(
        f"starting training loop, episodes: {ORIGINAL_RLEPSO_MAX_EPISODES}, "
        f"epochs: {ORIGINAL_RLEPSO_MAX_EPOCHS}"
    )
    ddpg.train(
        max_episodes=ORIGINAL_RLEPSO_MAX_EPISODES,
        max_epochs=ORIGINAL_RLEPSO_MAX_EPOCHS,
        max_steps=ORIGINAL_RLEPSO_MAX_STEPS,
        save_freq=ORIGINAL_RLEPSO_SAVE_FREQ,
    )


def test():
    pass


if __name__ == "__main__":
    logger.info('running ddpg.py directly')
    train()
