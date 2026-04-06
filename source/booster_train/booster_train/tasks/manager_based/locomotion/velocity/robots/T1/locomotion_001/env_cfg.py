from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster import BOOSTER_T1_CFG as ROBOT_CFG, T1_ACTION_SCALE
from booster_train.tasks.manager_based.locomotion.velocity.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from .locomotion_env_cfg import LocomotionEnvCfg

@configclass
class FlatEnvCfg(LocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T1_ACTION_SCALE
        
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        
        self.scene.terrain.terrain_type = "plane"