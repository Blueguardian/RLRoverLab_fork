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
    base = gym.make(task_name, cfg=cfg, viewport=video)
    # 1) enforce finite box
    base = FiniteActionBox(base)
    # 2) convert numpy→torch *before* any policy wrappers
    base = NumpyToTorchAction(base, device="cuda:0")

    # --- expert view: flatten → split → numpyify obs -----------------------
    def _expert():
        env = FlattenPolicyObs(base, TEACHER_KEYS + STUDENT_KEYS)
        env = TorchTensorToNumpy(env)
        env = NumpyToTorchAction(env)
        return env
    vec_expert = DummyVecEnv([_expert])

    # --- student view: flatten only, keep tensors --------------------------
    def _student():
        env = FlattenPolicyObs(base, TEACHER_KEYS + STUDENT_KEYS)
        env = StudentIdentity(env)
        env = NumpyToTorchAction(env)
        return env
    vec_student = DummyVecEnv([_student])

    # expose slices/shapes for feature-extractor
    vec_student.key_slices = _student().key_slices
    vec_student.key_shapes = _student().key_shapes

    vec_expert.key_slices = _expert().key_slices
    vec_expert.key_shapes = _expert().key_shapes

    return vec_student, vec_expert
