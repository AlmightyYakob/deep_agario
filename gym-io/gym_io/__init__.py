from gym.envs.registration import register

register(id="agario-v0", entry_point="gym_io.envs:AgarIOEnv")
register(id="slitherio-v0", entry_point="gym_io.envs:SlitherIOEnv")
