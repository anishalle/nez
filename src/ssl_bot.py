"""
SSL Bot — Staged Self-Learning Rocket League bot.

Training-ready bot with 4 progressive training stages:
  Stage 1 (Early):   Learn to touch the ball, don't forget jumping
  Stage 2 (Scoring): Introduce goal rewards, better touches
  Stage 3 (Middle):  Aerials, boost management, alignment
  Stage 4 (Later):   Polish with reduced LR, let it cook

Usage:
  python src/ssl_bot.py --stage 1          # Fresh training (default)
  python src/ssl_bot.py --stage 2          # Switch to scoring rewards
  python src/ssl_bot.py --stage 3          # Middle stage
  python src/ssl_bot.py --stage 4          # Later stage (reduced LR)
  python src/ssl_bot.py --stage 2 --checkpoint path/to/checkpoint  # Load specific checkpoint
"""

import argparse
import numpy as np
from math import sqrt

from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.common_values import (
    CEILING_Z, CAR_MAX_SPEED, BALL_MAX_SPEED, BLUE_TEAM,
    BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM,
)
from rlgym_ppo.util import MetricsLogger


# ---------------------------------------------------------------------------
# Custom Reward Functions
# ---------------------------------------------------------------------------

class SpeedTowardBallReward(RewardFunction):
    """Rewards player for moving toward the ball (normalized scalar projection)."""

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        dist = np.linalg.norm(pos_diff)
        if dist == 0:
            return 0.0
        direction = pos_diff / dist
        vel = player.car_data.linear_velocity / CAR_MAX_SPEED
        return float(np.dot(direction, vel))


class AirReward(RewardFunction):
    """Small reward for being airborne — prevents the bot from forgetting to jump."""

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0 if player.on_ground else 1.0


class StrongTouchReward(RewardFunction):
    """Rewards touching the ball, scaled by how much the ball's velocity changed.

    A slight push gives almost no reward; a strong shot gives lots.
    """

    def __init__(self):
        super().__init__()
        self.last_ball_vel = None

    def reset(self, initial_state: GameState):
        self.last_ball_vel = initial_state.ball.linear_velocity.copy()

    def pre_step(self, state: GameState):
        # Called once per step before get_reward — we store *previous* ball vel here
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        cur_vel = state.ball.linear_velocity
        if player.ball_touched and self.last_ball_vel is not None:
            delta = np.linalg.norm(cur_vel - self.last_ball_vel) / BALL_MAX_SPEED
            self.last_ball_vel = cur_vel.copy()
            return float(delta)
        self.last_ball_vel = cur_vel.copy()
        return 0.0


class AirTouchReward(RewardFunction):
    """Rewards touching the ball in the air, scaled by ball height and player air time.

    Uses a simple frame counter since PlayerData doesn't expose air_time directly.
    """

    MAX_AIR_FRAMES = 263  # ~1.75s at 120Hz / tick_skip=8 ≈ 26.25 steps/s => ~46 steps

    def __init__(self):
        super().__init__()
        self.air_frames: dict[int, int] = {}

    def reset(self, initial_state: GameState):
        self.air_frames = {}
        for p in initial_state.players:
            self.air_frames[p.car_id] = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.on_ground:
            self.air_frames[player.car_id] = 0
        else:
            self.air_frames[player.car_id] = self.air_frames.get(player.car_id, 0) + 1

        if not player.ball_touched:
            return 0.0

        air_time_frac = min(self.air_frames[player.car_id], self.MAX_AIR_FRAMES) / self.MAX_AIR_FRAMES
        height_frac = state.ball.position[2] / CEILING_Z
        return float(min(air_time_frac, height_frac))


class AggressionBiasGoalReward(RewardFunction):
    """Goal/concede reward with aggression bias.

    Concede penalty is reduced by `aggression_bias` fraction to promote aggressive play.
    e.g. aggression_bias=0.2 means concede penalty is 80% of goal reward.
    """

    def __init__(self, aggression_bias: float = 0.2):
        super().__init__()
        self.aggression_bias = aggression_bias
        self.last_scores: dict[int, tuple[int, int]] = {}

    def reset(self, initial_state: GameState):
        self.last_scores = {}
        for player in initial_state.players:
            if player.team_num == BLUE_TEAM:
                self.last_scores[player.car_id] = (initial_state.blue_score, initial_state.orange_score)
            else:
                self.last_scores[player.car_id] = (initial_state.orange_score, initial_state.blue_score)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            team_score, opp_score = state.blue_score, state.orange_score
        else:
            team_score, opp_score = state.orange_score, state.blue_score

        last_team, last_opp = self.last_scores.get(player.car_id, (team_score, opp_score))
        self.last_scores[player.car_id] = (team_score, opp_score)

        reward = 0.0
        if team_score > last_team:
            reward += 1.0
        if opp_score > last_opp:
            reward -= (1.0 - self.aggression_bias)
        return reward


# ---------------------------------------------------------------------------
# Metrics Logger
# ---------------------------------------------------------------------------

class SSLLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        p = game_state.players[0]
        return np.array([
            p.car_data.linear_velocity,
            game_state.blue_score,
            game_state.orange_score,
            p.boost_amount,
            0.0 if p.on_ground else 1.0,
            1.0 if p.ball_touched else 0.0,
        ], dtype=object)

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        total = len(collected_metrics)
        if total == 0:
            return

        avg_vel = np.zeros(3)
        total_boost = 0.0
        total_air = 0.0
        total_touches = 0.0
        blue_goals = 0
        orange_goals = 0

        for m in collected_metrics:
            avg_vel += m[0]
            blue_goals = max(blue_goals, m[1])
            orange_goals = max(orange_goals, m[2])
            total_boost += m[3]
            total_air += m[4]
            total_touches += m[5]

        avg_vel /= total
        wandb_run.log({
            "x_vel": avg_vel[0],
            "y_vel": avg_vel[1],
            "z_vel": avg_vel[2],
            "avg_boost": total_boost / total,
            "air_fraction": total_air / total,
            "touch_rate": total_touches / total,
            "blue_goals": blue_goals,
            "orange_goals": orange_goals,
            "cumulative_timesteps": cumulative_timesteps,
        })


# ---------------------------------------------------------------------------
# Stage Definitions
# ---------------------------------------------------------------------------

def get_stage_config(stage: int):
    """Returns (rewards_and_weights, learning_rate, entropy_coef) for a training stage."""
    from rlgym_sim.utils.reward_functions.common_rewards import (
        VelocityBallToGoalReward, FaceBallReward, EventReward,
        SaveBoostReward, AlignBallGoal,
    )

    if stage == 1:
        # Early: learn to touch the ball, don't forget jumping
        rewards = [
            (EventReward(touch=1), 50),
            (SpeedTowardBallReward(), 5),
            (FaceBallReward(), 1),
            (AirReward(), 0.15),
        ]
        lr = 2e-4
        ent = 0.01

    elif stage == 2:
        # Learning to score: introduce goal/ball-to-goal rewards
        rewards = [
            (StrongTouchReward(), 3),
            (VelocityBallToGoalReward(), 5),
            (SpeedTowardBallReward(), 2),
            (FaceBallReward(), 0.5),
            (AggressionBiasGoalReward(aggression_bias=0.2), 20),
            (AirReward(), 0.1),
        ]
        lr = 1e-4
        ent = 0.005

    elif stage == 3:
        # Middle: aerials, boost management, alignment
        rewards = [
            (StrongTouchReward(), 3),
            (AirTouchReward(), 5),
            (VelocityBallToGoalReward(), 5),
            (SpeedTowardBallReward(), 1),
            (SaveBoostReward(), 1),
            (EventReward(boost_pickup=0.5), 1),
            (AggressionBiasGoalReward(aggression_bias=0.2), 20),
            (AlignBallGoal(), 0.5),
        ]
        lr = 1e-4
        ent = 0.005

    elif stage == 4:
        # Later: same as stage 3 but lower LR — let it cook
        rewards = [
            (StrongTouchReward(), 3),
            (AirTouchReward(), 5),
            (VelocityBallToGoalReward(), 5),
            (SpeedTowardBallReward(), 1),
            (SaveBoostReward(), 1),
            (EventReward(boost_pickup=0.5), 1),
            (AggressionBiasGoalReward(aggression_bias=0.2), 20),
            (AlignBallGoal(), 0.5),
        ]
        lr = 5e-5
        ent = 0.001

    else:
        raise ValueError(f"Unknown stage: {stage}. Use 1-4.")

    return rewards, lr, ent


# ---------------------------------------------------------------------------
# Environment Builder
# ---------------------------------------------------------------------------

# Set by main() before Learner is created; read by build_rocketsim_env in each worker.
_CURRENT_STAGE = 1


def build_rocketsim_env():
    """Top-level env factory (no args) — must be picklable for multiprocessing."""
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        NoTouchTimeoutCondition, GoalScoredCondition,
    )
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import ContinuousAction

    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    rewards_and_weights, _, _ = get_stage_config(_CURRENT_STAGE)
    reward_fns = [r for r, _ in rewards_and_weights]
    reward_wts = [w for _, w in rewards_and_weights]

    env = rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=1,
        spawn_opponents=True,
        terminal_conditions=[NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()],
        reward_fn=CombinedReward(reward_functions=reward_fns, reward_weights=reward_wts), # pyright: ignore[reportArgumentType]
        obs_builder=DefaultObs(
            pos_coef=np.asarray([ # pyright: ignore[reportArgumentType]
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        ),
        action_parser=ContinuousAction(),
    )
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SSL Bot — Staged Self-Learning RL bot")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Training stage (1=early, 2=scoring, 3=middle, 4=later)")
    parser.add_argument("--checkpoint", type=str, default="latest",
                        help="Path to checkpoint directory to resume from (default: 'latest')")
    parser.add_argument("--n-proc", type=int, default=32,
                        help="Number of parallel environment processes")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    args = parser.parse_args()

    _, lr, ent_coef = get_stage_config(args.stage)

    print(f"=== SSL Bot — Stage {args.stage} ===")
    print(f"  Learning rate: {lr}")
    print(f"  Entropy coef:  {ent_coef}")
    print(f"  Processes:     {args.n_proc}")
    if args.checkpoint:
        print(f"  Checkpoint:    {args.checkpoint}")
    print()

    global _CURRENT_STAGE
    _CURRENT_STAGE = args.stage

    # Patch KBHit so it doesn't crash in non-interactive environments (e.g. SLURM)
    import sys
    if not sys.stdin.isatty():
        class _DummyKBHit:
            def set_normal_term(self): pass
            def getch(self): return ''
            def getarrow(self): return 0
            def kbhit(self): return False

        import rlgym_ppo.util.kbhit as _kbhit_mod
        _kbhit_mod.KBHit = _DummyKBHit
        import rlgym_ppo.util
        rlgym_ppo.util.KBHit = _DummyKBHit

    from rlgym_ppo import Learner
    # Patch the already-imported name in the learner module
    if not sys.stdin.isatty():
        from rlgym_ppo import learner as _learner_mod
        _learner_mod.KBHit = _DummyKBHit

    n_proc = args.n_proc
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        build_rocketsim_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=SSLLogger(),
        ppo_batch_size=50_000,
        ts_per_iteration=50_000,
        exp_buffer_size=150_000,
        ppo_minibatch_size=50_000,
        ppo_ent_coef=ent_coef,
        policy_lr=lr,
        critic_lr=lr,
        ppo_epochs=1,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=100_000,
        timestep_limit=1_000_000_000,
        log_to_wandb=not args.no_wandb,
        checkpoint_load_folder=args.checkpoint,
        policy_layer_sizes=(512, 512),
        critic_layer_sizes=(512, 512),
    )
    learner.learn()


if __name__ == "__main__":
    main()
