# Reward Table

| Event | Condition | Reward |
|---|---|---:|
| Step | Every environment step | -0.01 |
| Progress | Robot moves closer to target | +0.1 × distance improvement |
| Delivery | Robot successfully delivers the assigned task | +10.0 |
| Collision | Robot collides with another robot/obstacle | -10.0 |
| Danger | Robot enters/attempts to enter a dangerous area | -20.0 |

## Reward Formula

For each step:

Reward =
Step Penalty
+ Progress Reward
+ Delivery Reward
+ Collision Penalty
+ Danger Penalty

Where:

Progress =
Previous Distance to Target - Current Distance to Target