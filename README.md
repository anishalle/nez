  ┌───────┬──────────────────────────┬──────┬──────────────────────────────────────────────────────────┐
  │ Stage │          Focus           │  LR  │                       Key Rewards                        │
  ├───────┼──────────────────────────┼──────┼──────────────────────────────────────────────────────────┤
  │ 1     │ Touch ball, keep jumping │ 2e-4 │ EventReward(touch), SpeedTowardBall, FaceBall, AirReward │
  ├───────┼──────────────────────────┼──────┼──────────────────────────────────────────────────────────┤
  │ 2     │ Learn to score           │ 1e-4 │ StrongTouch, VelocityBallToGoal, AggressionBiasGoal      │
  ├───────┼──────────────────────────┼──────┼──────────────────────────────────────────────────────────┤
  │ 3     │ Aerials, boost mgmt      │ 1e-4 │ + AirTouch, SaveBoost, AlignBallGoal, boost_pickup       │
  ├───────┼──────────────────────────┼──────┼──────────────────────────────────────────────────────────┤
  │ 4     │ Polish, let it cook      │ 5e-5 │ Same as 3, reduced LR                                    │
  └───────┴──────────────────────────┴──────┴──────────────────────────────────────────────────────────┘

⏺ Stage-by-Stage Guide

  Stage 1 (Early — Touch the Ball)

  Move on when: touch_rate is high and stable (plateaued for ~10M steps)
  Typical duration: 30-80M steps
  Watch for:
  - Entropy dropping below 0.5 — bot is becoming too rigid, move on
  - Bot stops jumping — increase AirReward weight
  - Reward plateaus but touches are low — something is wrong with reward signal

  Stage 2 (Learning to Score)

  Move on when: bot consistently pushes ball into net, blue_goals climbing on W&B
  Typical duration: 50-150M steps
  Watch for:
  - Passive play — bot retreats to goal instead of attacking. Increase aggression_bias or decrease concede penalty
  - Clip fraction > 0.25 — updates too aggressive, consider lowering LR
  - Clip fraction < 0.05 — updates too small, learning stalled
  - Reward going up but goals not — bot is farming rewards (touching ball repeatedly without aiming)

  Stage 3 (Middle — Aerials & Boost)

  Move on when: bot does basic aerials, manages boost, saves and shoots
  Typical duration: 100-300M steps
  Watch for:
  - air_fraction not increasing — bot ignoring aerial rewards, increase AirTouchReward weight
  - avg_boost very low — bot wastes boost, increase SaveBoostReward
  - avg_boost very high — bot hoards boost, decrease SaveBoostReward
  - KL divergence spikes > 0.05 — training unstable, lower LR

  Stage 4 (Later — Let It Cook)

  No clear endpoint — improvement is slow and incremental
  Typical duration: 300M+ steps
  Watch for:
  - Reward going up but gameplay getting worse — rewards are being farmed, revisit reward design
  - Entropy near 0 — bot is fully deterministic, may be stuck in local optimum
  - Value loss increasing — critic can't keep up, may need more epochs or larger network

  General Red Flags (Any Stage)

  - KL divergence > 0.05 — lower LR
  - Clip fraction > 0.3 — lower LR or increase batch size
  - Clip fraction near 0 — LR too low or training stalled
  - Value loss exploding — return scale changed too fast, check reward weights
  - Entropy collapse — add more entropy coef or move to next stage
  - Reward oscillating wildly — reward function conflict, simplify rewards
