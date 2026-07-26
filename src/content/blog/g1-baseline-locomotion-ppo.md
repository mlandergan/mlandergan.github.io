---
title: "Reinforcement Learning for Humanoid Locomotion: A PPO Baseline for the Unitree G1"
description: "A from-scratch, cloud-reproducible baseline for training a Unitree G1 to walk with PPO in Isaac Lab: actor/critic design, reward shaping from first principles, and full Google Cloud setup."
date: 2026-07-25
draft: false
---

*Post 1 of a series on reinforcement-learning-based whole-body control for the Unitree G1. Code:
[link to this repo]. This post is scoped to the baseline walking policy; motion priors,
curriculum terrain, and manipulation are covered in later posts.*

The published literature on RL for humanoid whole-body control moves fast and assumes a lot:
dense trajectory-optimization notation, familiarity with a half-dozen papers' worth of reward and
observation conventions, and results that are hard to reproduce without the authors' exact
simulation setup. This series is meant as a primer — get a working baseline, a reward-authoring
workflow, and a training/deployment pipeline in your hands first, then use that as the reference
point for reading (and eventually reimplementing) papers like massively parallel legged RL
training ([Rudin et al., 2021](https://arxiv.org/abs/2109.11978)), adversarial motion priors
([Peng et al., 2021](https://arxiv.org/abs/2104.02180)), real-world humanoid locomotion with RL
([Radosavovic et al., 2024](https://arxiv.org/abs/2303.03381)), and expressive whole-body
imitation control ([Cheng et al., 2024](https://arxiv.org/abs/2402.16796)). None of those are
required reading for this post — they're where the series is headed.

## Why Reinforcement Learning for Humanoid Control

The classical alternative to a learned controller is model predictive control: at every control
step, solve a constrained trajectory optimization over a receding horizon, using an explicit
dynamics model and contact schedule. For a humanoid, that optimization is high-dimensional and
non-convex — 20+ actuated DoF, contact switching between feet, floor, and (eventually) hands and
objects — and solving it at the 100 Hz–1 kHz rates locomotion actually needs is a hard real-time
computing problem in its own right. Whole-body MPC on humanoids is an active research area
precisely because closing that loop fast enough usually means falling back to a reduced-order
model (centroidal dynamics, simplified contact assumptions) that trades away exactly the fidelity
you wanted in the first place.

RL moves that cost offline. Training is expensive — hours of simulation across thousands of
parallel environments — but the artifact that comes out the other end is a small MLP, and running
it at inference time is a single forward pass: sub-millisecond, fixed cost, no solver and no
explicit dynamics model on the critical path. Whatever robustness the policy has to contact
transitions, external pushes, or new terrain came from the range of situations it saw during
training, not from a model that had to anticipate each one analytically. The cost shows up on the
other side: MPC can retarget to a new objective by editing a cost function, no retraining
required, and it gives constraint-satisfaction guarantees an RL policy doesn't. An RL policy
generally needs new reward terms and another training run to pick up a new task.

That retraining cost is what the rest of this series is built to spend once. The sim environment,
the reward-authoring workflow, the PPO training loop, and the cloud pipeline in this post get
reused directly in later posts, not rebuilt: stair climbing adds a terrain curriculum on top of
this same setup, and loco-manipulation — picking up a box, opening a door — adds contact-rich
reward terms and an object in the scene. Classical whole-body control has to work out a new
contact-mode sequence and constraint set for each of those by hand; here it's mostly reward and
environment configuration. Getting a *stable, reasonably efficient* walking gait out of PPO, from
a small and interpretable reward, is the first of those configurations, and everything later in
the series depends on it working.

## What the algorithm is doing

We use PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) via
[rsl_rl](https://github.com/leggedrobotics/rsl_rl), the RL library Isaac Lab is built around. In
one sentence: thousands of G1 copies run in parallel in simulation, each collects short rollouts,
and an actor-critic network pair is updated on-policy with a clipped surrogate objective that
keeps each update close to the previous policy — which is what makes PPO comparatively stable and
easy to tune compared to naive policy gradient methods.

**Actor vs. critic: who sees what.** "Actor-critic" means we train two separate networks, and the
distinction between them is one of the more important ideas in this whole series, so it's worth
being precise about it. The actor is the policy: the network that maps an observation to an
action and gets deployed — and, critically, it can only be given observations that a *real* G1
could actually produce on hardware: base linear/angular velocity and orientation
derived from the IMU, joint positions/velocities from the joint encoders, the commanded walking
velocity (from a joystick or a higher-level planner), and its own previous action. Isaac Lab adds
sensor noise to these during training specifically so the actor doesn't learn to rely on
simulation-perfect signals it won't have on the real robot. The critic, by contrast, is a value
function, `V(s)`, used only to compute the advantage estimates PPO needs for its update (GAE). It
never runs on the real robot and is thrown away entirely once training finishes — only the actor
gets deployed. Because it has no sim-to-real obligation, the critic is allowed to see privileged,
"god mode" information that only exists inside the simulator: exact (noise-free) base velocity
instead of an IMU estimate, ground-truth contact forces at each foot, the exact
friction/mass/motor-strength randomization values applied to that particular training episode, or
the magnitude of a random push the environment just applied to the torso. Giving the critic
easier, lower-variance information to value a state from generally makes training faster and more
stable, even though the actor is still stuck doing the hard part — acting well from noisy,
partial, real-world-realistic observations. This pattern is usually called **asymmetric
actor-critic**.

<p align="center"><img src="/images/g1-baseline-locomotion/actor_critic.svg" width="640" alt="Actor observes only onboard-realistic, noisy signals and is what gets deployed; critic additionally sees privileged sim-only state and is discarded after training"></p>

*What the actor sees (onboard-realistic, noisy) versus what the critic is additionally allowed to
see (privileged simulator state) — only the actor's weights survive to deployment.*

Isaac Lab's manager-based environments support this directly: you can define a second observation
group (conventionally named `"critic"`) alongside the actor's `"policy"` group, and rsl_rl's PPO
runner will route the extra "privileged" fields only to the value network. This Project 1
baseline doesn't use that yet — both the stock G1 task and our minimal-reward variant currently
give the critic the exact same observations as the actor, so there's no asymmetric information
split in play here. That's a deliberate scoping choice for a "keep it simple" first post; it
becomes much more valuable in Post 4 (rough terrain / curriculum), where a privileged heightmap
and exact contact state are genuinely useful to a critic evaluating a state the actor can only
partially sense.

**From network output to torque.** The actor is a small feedforward MLP (`[256, 128, 128]`, ELU
activations). Its output is **not** a torque. It's one target value per actuated joint,
interpreted as a position offset from that joint's default pose. Those position targets are then
handed to an independent PD (proportional-derivative) controller *per joint*, running inside
PhysX at a much tighter loop rate than the policy itself:

<p align="center"><img src="/images/g1-baseline-locomotion/mlp_pd_pipeline.svg" width="460" alt="From network output to torque: the MLP feeds per-joint position targets into independent PD controllers, which produce torques applied in the physics step"></p>

*Actor MLP → per-joint position target → per-joint PD controller → torque → physics step, looped
once per control step.*

This is the standard choice for legged-robot RL: outputting position targets instead of raw
torques means the policy never has to learn low-level torque-tracking
dynamics — the PD gains (`Kp`, `Kd`) are set per joint to match the real actuators, independent of
the learned network — and it's exactly the control interface the real G1's joint drivers already
expect, which is a big part of why sim-trained locomotion policies transfer to hardware at all.

**Where each joint lives on the G1.** The reward terms in the next section reference specific
joints by name, so it helps to have the layout in mind. Unitree's own kinematic-frame render below
shows every actuated joint as a set of colored rotation axes (each tick is one degree of freedom)
on their 43-DOF G1 variant:

<p align="center"><img src="/images/g1-baseline-locomotion/g1_unitree_43dof_render.png" width="380" alt="Unitree's official G1 kinematic-frame render, showing each joint as a set of colored rotation axes"></p>

*Unitree's G1 kinematic-frame render, from the official developer documentation —
[support.unitree.com/home/en/G1_developer](https://support.unitree.com/home/en/G1_developer).*

It doesn't print joint names, but the axis clusters map onto familiar anatomy: shoulder (pitch,
roll, yaw), elbow (pitch, roll), a waist joint between chest and pelvis, then hip (yaw, roll,
pitch), knee, and ankle (pitch, roll) down each leg, plus the hand actuators. The joint *names*
this repo's code actually uses come from Isaac Lab's `G1_MINIMAL_CFG` asset — e.g. `hip_yaw_joint`,
`knee_joint`, `ankle_pitch_joint`.[^1]

## What changed in the environment and reward

Isaac Lab already ships a mature G1 flat-ground velocity task (`Isaac-Velocity-Flat-G1-v0`) with
~15 reward terms, tuned by people who iterated on this for a while. Copying it verbatim would
teach you nothing about *why* each term exists — so instead, we rebuild it from a small starting
point and let the *failures* motivate each addition.

**The reward we start with.** `Isaac-Velocity-Flat-G1-Baseline-Minimal-v0` (this repo's
`flat_env_cfg.py`) subclasses stock G1 and strips it down to six pedagogical terms: velocity
tracking (`track_lin_vel_xy_exp`, `track_ang_vel_z_exp` — the actual task), upright posture
(`flat_orientation_l2`, penalizing the torso tilting off vertical), base height (`base_height_l2`,
*added* — stock G1 has no such term[^2]), energy/torque penalty (`dof_torques_l2`) and joint
regularization (`dof_acc_l2`, `action_rate_l2` — discouraging jerky, high-torque solutions PPO
will happily find), and a foot slip penalty (`feet_slide`). Three more terms come along by
inheritance and stay active without being part of that pedagogical set: a small vertical-velocity
penalty (`lin_vel_z_l2`, -0.2), a small roll/pitch angular-velocity penalty (`ang_vel_xy_l2`,
-0.05), and a -200 penalty on episode termination (`termination_penalty`) — falling is still
expensive even in the "minimal" config. Run `reward_debug.py --list-tags` after training and
expect 11 active terms, not 6.

**Train it — here's what goes wrong.** Run this first (`TASK=Isaac-Velocity-Flat-G1-Baseline-Minimal-v0 ./scripts/train_walk.sh`) and
watch the rollout videos once it converges on the tracking task. Expect a policy that walks — the
six terms above are enough to produce forward locomotion that tracks the velocity command — but
with specific, predictable rough edges, each traceable to a term that *isn't* in this reward yet:

| What you'll see | Why | The term stock G1 uses to fix it |
|---|---|---|
| Arms swing wildly / flail | nothing penalizes arm joints drifting from a natural resting pose | `joint_deviation_arms` |
| Fingers twitch or spasm | same issue, on the hand actuators | `joint_deviation_fingers` |
| Torso/waist rotates oddly off-axis | nothing keeps the waist near its default yaw/roll/pitch | `joint_deviation_torso` |
| Hips splay outward, duck-footed gait | hip yaw/roll are free to drift | `joint_deviation_hip` |
| Irregular, shuffling gait; one leg barely leaves the ground | no shaping toward a swing/stance rhythm | `feet_air_time` |
| Occasional stiff, jarring ankle motion at range limits | nothing discourages hitting ankle joint limits | `dof_pos_limits` |

The minimal reward isn't wrong — it optimizes exactly what it's given. A stable gait and a
natural-looking one are different optimization targets, and closing that gap here just takes a
handful of boring regularization terms, not new algorithms.

**Adding the terms back.** Add these six terms back in, and you've essentially reconstructed Isaac Lab's stock
`G1FlatEnvCfg` from first principles:

| Term | Weight | Scope |
|---|---:|---|
| `joint_deviation_hip` | -0.1 | hip yaw/roll only (pitch stays free — it's essential for stepping) |
| `joint_deviation_arms` | -0.1 | shoulder pitch/roll/yaw, elbow pitch/roll |
| `joint_deviation_fingers` | -0.05 | finger joints |
| `joint_deviation_torso` | -0.1 | torso/waist joint |
| `feet_air_time` | +0.75 | swing-phase shaping, 0.4s threshold |
| `dof_pos_limits` | -1.0 | ankle pitch/roll only |

That's already implemented and registered as `Isaac-Velocity-Flat-G1-v0`, so there's nothing new
to write; just run it. Here's that reconstruction as Isaac Lab actually wrote it —
`config/g1/flat_env_cfg.py` in full:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
```

Notice what this file *doesn't* do: it never defines a reward function. `G1FlatEnvCfg` subclasses
`G1RoughEnvCfg`, which is where the actual `RewTerm` objects live — function, default weight, and
params — one per term (`track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp,
weight=1.0, ...)`, and so on). `flat_env_cfg.py` only *reweights* the handful of terms that need
to change once you drop the height scanner and terrain curriculum for flat ground —
`self.rewards.<term_name>.weight = <value>`, called after `super().__post_init__()` so the parent
class's rewards object already exists to edit. `feet_air_time.params["threshold"] = 0.4` shows
params are editable the same way, not just weights. This repo's own `flat_env_cfg.py` uses the
identical pattern for the minimal variant — subclass, call `super().__post_init__()` first, edit
`self.rewards.*` in place. The trailing `# Commands` block is unrelated to rewards: it widens the
velocity-command sampling range now that the robot isn't fighting rough terrain — `lin_vel_y`
opens up to ±0.5 m/s, where rough terrain keeps it locked to 0.

**Retrain and compare.** `TASK=Isaac-Velocity-Flat-G1-v0 ./scripts/train_walk.sh` — same PPO hyperparameters, same number
of environments, only the reward differs, so any change in the resulting gait is attributable to
the reward terms, not the optimizer. Expect calmer arms, stiller fingers, straighter hips, more
regular step cadence, and no ankle-limit slamming, relative to the minimal run.

## Setting up the cloud GPU: create, connect, train, visualize

### Why a Spot T4

Isaac Sim's own docs list `n1-standard-8` + one `nvidia-tesla-t4` as the documented minimum spec
for running Isaac Sim on Google Cloud — it's the cheapest GPU shape NVIDIA explicitly supports
for it. Add GCP's **Spot** provisioning (a steep discount off on-demand, in exchange for the VM
being preemptible) on top of that. Training checkpoints every 50 iterations, so a Spot preemption
costs at most a few minutes of progress if you resume from the last checkpoint afterward. GPU and
vCPU pricing changes over time and by region — check current numbers yourself with the
[GCP pricing calculator](https://cloud.google.com/products/calculator) before committing to a long
run.

### 1. Create the VM

```bash
gcloud compute instances create g1-baseline-t4 \
  --project=<your-project-id> \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100 \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --metadata="install-nvidia-driver=True"
```

`--instance-termination-action=STOP` means a Spot preemption stops the VM (keeping its disk)
instead of deleting it — restart it with `gcloud compute instances start g1-baseline-t4
--zone=us-central1-a` and resume from the last checkpoint.

(`scripts/gcp_create_vm.sh` wraps this command; `PROJECT_ID`, `ZONE`, `INSTANCE_NAME`,
`MACHINE_TYPE`, `ACCELERATOR`, and `BOOT_DISK_SIZE_GB` are all overridable via environment
variables.)

### 2. Connect

```bash
gcloud compute ssh g1-baseline-t4 --zone=us-central1-a
```

To view live training curves later without opening a second connection, forward TensorBoard's
port over the same SSH session instead:

```bash
gcloud compute ssh g1-baseline-t4 --zone=us-central1-a -- -L 6006:localhost:6006
```

### 3. One-time environment setup (on the VM)

Clone this repo, install Docker + the NVIDIA Container Toolkit, and build the training image:

```bash
git clone <this repo> && cd RL_WBC_G1

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # log out and re-SSH after this

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

nvidia-smi   # sanity check: driver sees the GPU

# NGC login -- needs a free account + API key from https://ngc.nvidia.com
# username is the literal string "$oauthtoken", password is your API key
docker login nvcr.io

docker build -f docker/Dockerfile -t rl-wbc-g1-baseline .
docker run --rm --gpus all rl-wbc-g1-baseline nvidia-smi   # confirm the GPU is visible in-container
```

(`scripts/remote_setup.sh` in this repo automates this whole block. The build also runs
`docker/patch_register_task.py`, which hooks our custom task's `gym.register()` call into Isaac
Lab's `train.py`/`play.py` right after Isaac Lab registers its own built-in tasks — that's what
makes `Isaac-Velocity-Flat-G1-Baseline-Minimal-v0` resolve at all.)

### 4. Train

```bash
TASK=Isaac-Velocity-Flat-G1-Baseline-Minimal-v0 ./scripts/train_walk.sh   # minimal reward first
TASK=Isaac-Velocity-Flat-G1-v0 ./scripts/train_walk.sh                    # then stock, for comparison
```

Each run writes checkpoints and TensorBoard logs to `./logs/rsl_rl/<experiment_name>/<timestamp>/`
on the VM.

### 5. Visualize

**Live training curves**, through the SSH tunnel opened in step 2:

```bash
tensorboard --logdir logs/rsl_rl --port 6006 --bind_all
```

then open `http://localhost:6006` in a browser on your local machine.

**Rollout video**, once a run has a checkpoint:

```bash
CHECKPOINT=logs/rsl_rl/g1_flat_baseline_minimal/<run>/model_1500.pt ./scripts/export_video.sh
```

saves an mp4 under `logs/rsl_rl/<experiment_name>/<run>/videos/play/`. For a rollout without the
video (faster, for a quick sanity check), `scripts/eval_policy.sh` takes the same `TASK` /
`CHECKPOINT` / `NUM_ENVS` variables and skips the `--video` flag.

**Per-reward-term diagnostics** — run directly on the VM against `./logs`, or later on your local
machine once logs are synced (step 6):

```bash
python isaaclab_project/g1_baseline/reward_debug.py \
  --logdir logs/rsl_rl/g1_flat_baseline_minimal --list-tags
python isaaclab_project/g1_baseline/reward_debug.py \
  --logdir logs/rsl_rl/g1_flat_baseline_minimal --tag-regex "Episode_Reward/.*" --out minimal.png
```

### 6. Pull results down and tear down

From your **local** machine (not the VM), use `scripts/sync_results.sh` rather than a raw `gcloud
compute scp` — the script deliberately targets the parent of `./logs` so that syncing twice (e.g.
once after the minimal run, again after the stock run) doesn't nest into `./logs/logs/...`:

```bash
PROJECT_ID=<your-project-id> ./scripts/sync_results.sh
```

then delete the VM so it stops billing:

```bash
gcloud compute instances delete g1-baseline-t4 --zone=us-central1-a
```

(`scripts/gcp_teardown_vm.sh` wraps the delete command above with a confirmation prompt, since
deleting the VM is destructive.)

## What to look for, and common failure modes

**In the rollout videos:** falls in the first few hundred iterations are normal — watch the trend
in episode length / fall rate over training, not any single early rollout. Beyond the specific
failure modes in the table above, a broadly useful sanity check is whether the policy actually
tracks the *commanded* velocity, or has converged on some fixed gait regardless of command — vary
the command range in the `_Play` config and check the robot responds.

**If training goes sideways:**

- **Policy learns to stand still.** Check the velocity-tracking term's weight relative to the
  penalty terms — including the always-on -200 `termination_penalty`, which makes falling
  expensive enough that standing still can look attractive by comparison. If standing still
  minimizes total penalty better than the tracking reward compensates for, that's what PPO will
  find.
- **Policy hops or vibrates in place.** Usually under-penalized `action_rate_l2` / `dof_acc_l2`,
  or `num_envs` too low, giving the critic noisy value estimates.
- **`base_height_l2` pins the robot in a bad crouch.** You almost certainly used the wrong
  `TARGET_BASE_HEIGHT_M`.[^2]
- **Training looks fine on TensorBoard, robot immediately falls in `play.py`.** Check that
  `enable_corruption`/observation noise is disabled in the `_PLAY` config (it is, by default,
  in both variants here) — training-time observation noise can mask a policy that's fragile to
  clean observations.

## Next step

Post 2 replaces the hand-written style regularization here (`joint_deviation_*`, `flat_orientation_l2`
tuning) with an Adversarial Motion Prior discriminator trained against a small reference-motion
clip. The PPO runner and cloud setup carry over unchanged; only the reward mechanism changes. The
comparison this post set up (minimal vs. reconstructed-stock reward) is exactly the gap AMP is
meant to close without hand-tuning a dozen regularization terms one failure mode at a time.

---

[^1]: Anatomically consistent with, but not an exact 1:1 index match to, Unitree's own numbering
    above — it has more axis ticks per arm than there are matching joint names in our config,
    since the finger actuators aren't broken out individually.
[^2]: That constant is a placeholder in `flat_env_cfg.py` and *must* be measured from a
    zero-action rollout of your own installed G1 asset — log the torso link's world-frame Z
    height a few steps after reset — rather than copied from this post.
