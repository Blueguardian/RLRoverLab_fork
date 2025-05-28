# rover_il/DAgger/build_envs.py
# --------------------------------------------------------------------- #
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from .wrappers import (
    FiniteActionBox, NumpyToTorchAction,
    FlattenPolicyObs,
    TorchTensorToNumpy, StudentIdentity,
    TEACHER_KEYS, STUDENT_KEYS,
)

def make_shared_vecenv(task_name, cfg, video=False):
    # ------------------------------------------------------------------ #
    #  Base simulator (shared)                                           #
    # ------------------------------------------------------------------ #
    base = gym.make(task_name, cfg=cfg, viewport=video)
    base = FiniteActionBox(base)
    base = NumpyToTorchAction(base, device="cuda:0")

    # ------------------------------------------------------------------ #
    #  Expert view: numpy observations                                    #
    # ------------------------------------------------------------------ #
    def _build_expert():
        env = FlattenPolicyObs(base, TEACHER_KEYS + STUDENT_KEYS)
        env = TorchTensorToNumpy(env)
        return env

    # ------------------------------------------------------------------ #
    #  Student view: torch observations                                   #
    # ------------------------------------------------------------------ #
    def _build_student():
        env = FlattenPolicyObs(base, TEACHER_KEYS + STUDENT_KEYS)
        env = StudentIdentity(env)
        return env

    vec_expert  = DummyVecEnv([_build_expert])
    vec_student = DummyVecEnv([_build_student])

    # expose authoritative mapping
    mapping_env           = vec_expert.envs[0]
    vec_expert.key_slices = mapping_env.key_slices
    vec_expert.key_shapes = mapping_env.key_shapes
    vec_student.key_slices = mapping_env.key_slices
    vec_student.key_shapes = mapping_env.key_shapes

    return vec_student, vec_expert
