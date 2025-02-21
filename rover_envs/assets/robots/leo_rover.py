import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from rover_envs.envs.navigation.utils.articulation.articulation import RoverArticulation

_LEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "leo_rover", "leo.usd")

LEO_ROVER_CFG = ArticulationCfg(
    class_type=RoverArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=_LEO_PATH,
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.04, rest_offset=0.01),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=0.4, # m/s
            max_angular_velocity=60.0, # deg/s
            max_depenetration_velocity=1.0, # m/s
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4)
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_vel={".*Drive_Joint": 0.0},
    ),
    actuators={
        "base_drive": ImplicitActuatorCfg(
            joint_names_expr=[".*Drive_Joint"],
            velocity_limit=6,
            effort_limit=12,
            stiffness=100.0,
            damping=4000.0,
        ),
        "passive_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*Bogie_Joint"],
            velocity_limit=10,
            effort_limit=0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
