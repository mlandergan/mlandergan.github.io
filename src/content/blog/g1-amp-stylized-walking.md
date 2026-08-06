---
title: "Stylized Walking for the Unitree G1 with Adversarial Motion Priors"
description: "Replacing hand-tuned style regularization with a learned motion prior: the AMP paper, where reference motion data actually comes from, and a velocity-tracking G1 policy trained with skrl."
date: 2026-08-02
draft: false
---

*Post 2 of a series on reinforcement-learning-based whole-body control for the Unitree G1. Code:
[github.com/mlandergan/rl-wbc-g1-amp](https://github.com/mlandergan/rl-wbc-g1-amp). Post 1
covered a velocity-tracking walk built from a six-term hand-tuned reward; this post replaces the
style terms in that reward with a learned motion prior.*

Unitree has published a demo of the G1 performing a martial-arts routine — spinning kicks,
punches, one-legged balances — with fluid, human-like timing, but hasn't published the training
method behind that specific video. Coverage of it describes a general recipe of
motion-capture-driven reinforcement learning in simulation followed by sim-to-real transfer. The
closest publicly documented technique for this category of skill on the G1 is
[KungfuBot](https://arxiv.org/abs/2506.12851) (Xie et al., 2025). It uses adaptive motion
tracking, a curriculum that adjusts tracking-error tolerance as training progresses while
imitating a specific, time-aligned reference clip. That's a different branch of the
motion-imitation family from Adversarial Motion Priors, the subject of this post, which don't
track a specific clip at all.

<div style="position:relative; padding-bottom:56.25%; height:0; overflow:hidden; max-width:100%; border-radius:0.5rem; border:1px solid #e0d9cf;">
  <iframe src="https://www.youtube.com/embed/O5GphCrjx98" style="position:absolute; top:0; left:0; width:100%; height:100%; border:0;" title="Unitree G1 Kungfu Kid V6.0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

## What Adversarial Motion Priors Are

AMP was introduced in
[Peng, Ma, Abbeel, Levine, and Kanazawa, "AMP: Adversarial Motion Priors for Stylized
Physics-Based Character Control"](https://arxiv.org/abs/2104.02180) (ACM Transactions on Graphics,
2021). From the abstract:

> Data-driven methods that leverage motion tracking are a prominent class of techniques for
> producing high fidelity motions for a wide range of behaviors. However, the effectiveness of
> these tracking-based methods often hinges on carefully designed objective functions, and when
> applied to large and diverse motion datasets, these methods require significant additional
> machinery to select the appropriate motion for the character to track in a given scenario. [...]
> These motion clips are used to train an adversarial motion prior, which specifies style-rewards
> for training the character through reinforcement learning (RL). The adversarial RL procedure
> automatically selects which motion to perform, dynamically interpolating and generalizing from
> the dataset.

The reward is split into two independent parts: a task reward that specifies *what* the character
should do, and a style reward that specifies *how*. The task reward is ordinary: the same kind of
hand-written velocity-tracking term used in Post 1. The style reward comes from a discriminator,
trained alongside the policy to distinguish transitions produced by the policy from transitions
sampled out of a reference motion dataset. The policy is trained to fool it. A transition the
discriminator cannot distinguish from the reference data earns a high style reward; one it can
easily flag as policy-generated earns a low one.

The discriminator does not track a specific clip. It scores a short transition against the
reference dataset's distribution as a whole:

> Unlike standard tracking objectives, which measure pose similarity with respect to a specific
> reference motion, the motion prior returns a general score indicating the similarity of the
> character's motion to the motions depicted in the dataset, without explicitly comparing to a
> particular motion clip.

No phase variable (a clock telling the policy where it should be in a specific reference clip),
no clip selection, no synchronization between the policy's current point in an episode and a
specific frame of reference motion. Those are exactly the requirements that make tracking-based
methods like KungfuBot's curriculum necessary in the first place.

<p align="center"><img src="/images/g1-amp-stylized-walking/amp_paper_fig2_overview.png" width="600" alt="Schematic overview of the AMP system: environment state feeds both the policy and the motion prior, whose style reward is added to the task reward and returned to the policy"></p>

*Figure 2 from [Peng et al., 2021](https://arxiv.org/abs/2104.02180), "AMP: Adversarial Motion
Priors for Stylized Physics-Based Character Control," reproduced here for commentary. The motion
prior scores the current state against the reference dataset to produce the style reward
$r^S_t$, added to the task reward $r^G_t$ to form the total reward $r_t$ the policy is trained on.*

## The Math, Term by Term

The total reward the policy optimizes is a weighted sum of the task reward and the style reward:

$$
r(s_t, a_t, s_{t+1}, g) = w^G\, r^G(s_t, a_t, s_{t+1}, g) + w^S\, r^S(s_t, s_{t+1})
$$

- $r^G$ — the task reward: what the character should accomplish, conditioned on a goal $g$ (in
  this project, a commanded base velocity, the same task reward from Post 1).
- $r^S$: the style reward — how the character should move, produced by the discriminator.
- $w^G$, $w^S$: fixed scalar weights balancing the two.
- $s_t$, $a_t$, $s_{t+1}$: the state, action, and next state at timestep $t$.

The discriminator $D$ is trained to separate reference transitions from policy transitions using a
least-squares GAN objective:

$$
\arg\min_D \; \mathbb{E}_{d^{\mathcal{M}}(s,s')}\left[(D(s,s') - 1)^2\right] +
\mathbb{E}_{d^{\pi}(s,s')}\left[(D(s,s') + 1)^2\right]
$$

- $\mathcal{M}$: the reference motion dataset; $d^{\mathcal{M}}(s,s')$ is the probability of
  observing transition $(s,s')$ in it.
- $\pi$: the current policy; $d^{\pi}(s,s')$ is the probability of observing that same transition
  under the policy's rollouts.
- The discriminator is pushed toward outputting $+1$ on reference transitions and $-1$ on
  policy-generated ones.

The paper chooses this least-squares formulation over the sigmoid cross-entropy loss used in
standard GAIL-style adversarial imitation (Generative Adversarial Imitation Learning, the older
GAN-based approach this method builds on), stating that the latter "tends to lead to optimization
challenges due to vanishing gradients as the sigmoid function saturates."

The style reward computed from the trained discriminator's output is:

$$
r^S(s_t, s_{t+1}) = \max\left[0,\; 1 - 0.25\left(D(s_t, s_{t+1}) - 1\right)^2\right]
$$

This clips the reward to $[0, 1]$ and peaks at $D = 1$, a transition the discriminator is fully
convinced came from the reference dataset.

The discriminator does not see raw simulation state. It sees a fixed feature map $\Phi(s_t)$
extracted from it. Per the paper, that includes the root's linear and angular velocity in the
character's local frame, and the local rotation and local velocity of each joint: all relative
quantities, none of them tied to a specific point in a specific reference clip.

## Where Reference Motion Comes From

AMP's style reward needs a dataset of motion clips, $\mathcal{M}$. For a legged robot, every
option in this space is a combination of two distinct stages, and keeping them separate makes
it much easier to see what you're actually choosing between:

1. **Original human motion** — a person moves, and that movement gets captured somewhere: a
   mocap studio, a wearable rig, or an ordinary video.
2. **Retargeting** — that human motion is mapped onto the robot's joint structure, which has
   different limb proportions, fewer degrees of freedom, and hard joint limits. A clip is not
   usable for training until this step is done.

Datasets and tools differ mainly in *which stages they hand you pre-made*. Some give you raw
human motion and leave retargeting to you; some give you fully retargeted robot trajectories;
some wrap the entire path from video to robot joints in one pipeline.

### Stage 1: capturing original human motion

**Marker-based mocap datasets.** The bulk of publicly available human motion was recorded in
capture studios with optical markers.
**AMASS** ([Archive of Motion Capture as Surface Shapes](https://amass.is.tue.mpg.de/), Max
Planck Institute for Intelligent Systems) aggregates dozens of these datasets and
re-parameterizes all of them onto a single body model, SMPL. It is large and diverse, and purely
*human* motion: using it means running your own retargeting pipeline (stage 2) first. **LAFAN1**,
Ubisoft's mocap dataset released for animation research, is another staple: smaller than AMASS,
but well-curated locomotion.

**Recording your own.** A manually operated capture system (optical marker-based, IMU-based, or
depth-camera-based) recording a human performer directly. This is how most of the mocap behind
AMASS and LAFAN1 was originally produced. It gives full control over the recorded motions at the
cost of hardware, a performer, and a capture session. The output is again human motion; stage 2
still applies.

**Ordinary video.** The newest and most scalable source: extracting 3D human motion from plain
video, no capture rig at all. The standard intermediate representation is
**[SMPL](https://smpl.is.tue.mpg.de/)** (Skinned Multi-Person Linear model), a parametric human
body model that represents a person's shape and pose with a small set of numbers, roughly a
few hundred parameters rather than a full 3D mesh, that can be fit to video frames using
existing pose-estimation models. A concrete example of the pose-estimation step:
**[PromptHMR](https://github.com/yufu-wang/PromptHMR)** (Wang et al., CVPR 2025) recovers SMPL
bodies from images or video using a prompt (a box, a mask, or a text description) to specify
which person in the scene to reconstruct, handling multi-person scenes and moving cameras.

<p align="center"><img src="/images/g1-amp-stylized-walking/prompthmr_teaser.jpg" width="700" alt="PromptHMR: an image plus a box, mask, or text prompt produces SMPL body meshes for the specified person or people"></p>

*From the PromptHMR project (Wang et al., CVPR 2025): image plus prompt in, SMPL body meshes out.*

SMPL is what makes video-sourced motion scalable: it is the common interchange format nearly
every video-to-3D-pose method already outputs, so a retargeting pipeline built once can consume
motion from many different upstream sources — studio mocap and internet video alike.

<p align="center"><img src="/images/g1-amp-stylized-walking/video2robot_pipeline.svg" width="640" alt="Pipeline: video, to 3D human pose estimation, to SMPL parameters, to retargeting, to robot joint trajectories"></p>

*The two stages in sequence for video: pose estimation produces SMPL parameters per frame
(stage 1), and retargeting maps those onto the target robot's joints (stage 2). AMASS and
LAFAN1 enter the same retargeting step from a different starting point: existing marker-based
mocap instead of video.*

### Stage 2: retargeting onto the robot

Retargeting maps a human skeleton's motion onto the robot's kinematic tree, typically by
numerical optimization against end-effector pose, joint position, and joint velocity
constraints, subject to the robot's joint limits. It is the step where "a person strutting"
becomes "a G1 joint trajectory," and it is where most of the practical friction lives:
the result is kinematically plausible but carries no guarantee of being dynamically consistent
for the robot's mass distribution and actuators.

You can run this stage yourself with a general-purpose tool like
**[GMR](https://github.com/YanjieZe/GMR)** (General Motion Retargeting), which ships presets for
common humanoids including the G1 — or skip it entirely by using a *pre-retargeted* dataset.
[`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
is exactly that: LAFAN1 already retargeted onto the Unitree G1 (and H1, H1_2), distributed as
per-frame CSVs of robot joint configurations.

**End to end in one pipeline.** **[video2robot](https://github.com/AIM-Intelligence/video2robot)**
(AIM-Intelligence) is a complete system that covers both stages and even generates the source
video itself: a text prompt or an existing video goes through a generative video model (Veo or
Sora), then PromptHMR for pose estimation (stage 1), then GMR for retargeting (stage 2),
producing motion for the Unitree G1, Unitree H1, and Booster T1. I have run this pipeline end to
end and it produces usable reference motion with no retargeting code of my own. One license note
carries through: PromptHMR is released for non-commercial research use only, a restriction
video2robot inherits.

## This Project's Task: Stylized Velocity Tracking

The task in this post is not pure imitation. The G1 is trained to track a commanded base
velocity, the same task reward from Post 1, and the style reward is added on top of it, not
swapped in for it.

The reference clip is one animation from [Mixamo](https://www.mixamo.com/#/?page=1&query=Strut%20Walking)'s
library, "Strut Walking," retargeted onto the G1's 29-DOF joint convention with
[GMR](https://github.com/YanjieZe/GMR). It's not a pre-retargeted dataset like LAFAN1. This
project did the retargeting itself. Mixamo restricts redistributing the raw motion asset, so
treat this the same way as the LAFAN1 caveat above: personal use, and check current terms
yourself before reusing it.

Post 1's joint-deviation and regularization terms are still in the reward here, running alongside
AMP, not replaced by it. The style reward shapes the gait; the regularization terms act as a
safety net under it. That combination turned out to matter more than expected.

The commanded velocity range is narrowed to roughly match the clip's own pace, about 0.83 m/s,
near straight ahead, rather than reusing Post 1's full command range — so task and style ask for
compatible things instead of pulling the policy in two directions at once.

One more split is deliberate: the policy's observation includes the velocity command, so it
knows what it's being asked to do, but the discriminator's observation never does. Otherwise
the discriminator could learn to reward "moving at the commanded speed" as a shortcut for
"looking like the reference clip," which isn't the same thing.

## Training with `skrl`

[`skrl`](https://skrl.readthedocs.io/) is an open-source reinforcement learning library for
PyTorch and JAX, with native support for Isaac Lab environments (NVIDIA's GPU-accelerated robot
simulation framework, covered in Post 1) and a built-in `AMP` agent class that trains the policy,
value, and discriminator networks together. That's the reason it's the training library for this
post instead of Post 1's `rsl_rl`, which has no AMP implementation for Isaac Lab. The environment
and task configuration layer stays the same style of Isaac Lab code as Post 1.

## The Code

**Retargeting, done once, offline.** Mixamo's "Strut Walking" clip is a human animation, not a
robot trajectory. [GMR](https://github.com/YanjieZe/GMR) retargets it onto the G1's skeleton and
writes out root position, root rotation, and joint angles per frame. Here's the human animation
next to the retargeted G1, same clip, same timing:

<video src="/videos/g1-amp-stylized-walking/retargeting_sidebyside.mp4" width="900" controls loop muted></video>

*Left: the original Mixamo animation. Right: GMR's retargeted G1, driven by the same motion.*

GMR's output only carries root pose and joint angles, no per-body Cartesian data. A conversion
script (`convert_gmr_to_npz.py`) runs MuJoCo forward kinematics on that trajectory to get every
body's position and rotation, finite-differences them for velocities, and writes the result into
Isaac Lab's own `.npz` motion schema. Isaac Lab's stock `MotionLoader` reads that file directly.
No custom loader was needed here: the conversion happens once, ahead of training, not on every
training step.

**Action space: a small nudge from standing, not a swing across the joint's range.** The policy
output is a per-joint offset from the robot's own default pose, tracked from there by a
proportional-derivative (PD) controller:

```python
def _apply_action(self):
    target = self.robot.data.default_joint_pos.clone()
    target[:, self.action_dof_indexes] += self.cfg.action_scale * self.actions
    self.robot.set_joint_position_target(target)
```

`action_scale` is 0.5, a small offset from the robot's own standing pose. The size of that offset
matters more than it looks: a large one lets a single step of exploration noise throw a joint
across its entire range of motion, producing a physically incoherent target almost every step,
independent of how good the policy is.

**Discriminator observation vs. policy observation.** Both start from the same feature vector:
joint positions, joint velocities, root height, root orientation, root linear and angular
velocity, and key body positions relative to the root.

```python
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
```

`quaternion_to_tangent_and_normal` converts the root orientation from a raw quaternion into two
rotated unit vectors instead, a representation with no discontinuities for the network to learn
around, unlike a quaternion's double-cover ambiguity.

The discriminator gets exactly this. The policy gets this plus the velocity command appended.
Same function, two different consumers, one of them sees more than the other on purpose.

**Training is config-driven.** There's no Python code that instantiates the AMP agent by hand.
Isaac Lab's own `train.py` reads a yaml config (model shapes, PPO [Proximal Policy Optimization]
hyperparameters, task/style reward weights, discriminator settings) and runs it through `skrl`'s
built-in `AMP` agent class. The weights that produced the policy in the Results section below:

```yaml
task_reward_weight: 0.5
style_reward_weight: 0.5
```

These are `skrl`'s names for $w^G$ and $w^S$ from the combination equation earlier in this post,
an even split between task and style. That's the AMP paper's own canonical starting point, and it
turned out to be the right call here too, but only once style had already been established: this
run was warm-started from a checkpoint trained on style reward alone, so the task reward wasn't
competing with style from a standing start. It was steering a policy that already knew how to
move like the reference clip. More on why in the Results section.

## Cloud and Training Setup

The Docker image and GCP setup from Post 1 carry over in spirit, not in exact shape: this project
runs on a `g2-standard-4` + `nvidia-l4` VM rather than Post 1's `n1-standard-8` + T4, since AMP's
larger networks and the discriminator's extra forward pass benefit from the newer GPU. Same
NVIDIA Container Toolkit install, same NGC login. The container installs `skrl` alongside
`rsl_rl` rather than in place of it, and the reference motion (the Mixamo clip retargeted through
GMR, see "The Code" above) is converted locally and baked into the image rather than downloaded
at container start.

## Results

This policy trained in two stages. First, pure imitation: style reward only, no task reward, to
learn what the reference motion looks like without a competing objective. Then 24,000 steps with
the task reward reintroduced at an even 0.5/0.5 split with style, warm-started from the first
stage's checkpoint.

Training the combined weights from scratch, no warm start, is a perfectly reasonable alternative.
Nothing about AMP requires this particular order. This project took the two-stage path because
getting style to work at all had already turned into its own investigation, so once a
pure-imitation checkpoint existed, continuing from it to add the task reward was the natural next
step.

<video src="/videos/g1-amp-stylized-walking/combined_frontcam.mp4" width="900" controls loop muted></video>

**Tensorboard Logs**
<p align="center">
<img src="/images/g1-amp-stylized-walking/results_mean_reward.png" width="900" alt="Mean total reward over training, climbing from about -175 to about 320 over 24,000 steps">
</p>

**Episode length** dips to about 60 steps in the first few hundred steps of this run. That's the
value function (the network that predicts expected future reward, used to guide training)
readjusting to a reward signal it hadn't seen during the imitation stage. It then climbs back to
299 out of 299 and holds there for the rest of training.

**Velocity tracking reward** starts around 0.59 (a walk paced to match the reference clip already
tracks a nearby commanded velocity reasonably well) and climbs to 0.87 by the end, out of a
maximum of 1.0.

<p align="center">
<img src="/images/g1-amp-stylized-walking/results_discriminator_loss.png" width="900" alt="Discriminator loss over training, opening near 2.0, dipping to about 0.85 around step 5,000, then climbing back past 2.0">
</p>

**Discriminator loss** opens around 2.0, dips to about 0.85 while the task reward is first
settling in (roughly steps 4,000 to 7,000, the same window the value function is readjusting in),
then climbs back past 2.0 and holds there for the rest of the run. A low loss means the
discriminator can easily tell policy motion from reference motion; a high one means it can't. This
is the number that actually separates a working AMP run from a broken one. A policy can track
velocity perfectly with discriminator loss near zero and still look nothing like the reference
clip. An earlier attempt at this exact task produced exactly that: correct velocity, crossed feet,
one arm pinned in place. The dip and recovery here are the task reward and the style reward briefly
competing, then settling into the same policy.



## Next Step

With task and style rewards both in place and a working discriminator, the natural next
comparison is an ablation across style weight: no AMP, a weak style reward, and a strong one,
against the same task reward and the same checkpoint budget, to show what the style term buys
over Post 1's hand-tuned regularization.

## Looking Ahead

A speculative note, outside the scope of this series: I expect frontier humanoid labs to keep
scaling motion-tracking data collection: more capture sessions, more video-to-SMPL pipelines like
the one described above, broader coverage of the human motion space. And I expect them to train
whole-body controllers on a combination of simulated task-reward fine-tuning and large-scale
motion tracking, rather than committing to one over the other. The same reasoning that motivates blending AMP with
a task reward in this post holds at larger scale: a style prior alone does not guarantee task
success, and a task reward alone does not guarantee robustness across scenarios the training
distribution did not anticipate.

This shape already exists in at least one deployed system. Figure AI's
[Helix 02](https://www.figure.ai/news/helix-02) splits control into three layers running at
different timescales: S2, semantic reasoning at low frequency (scene understanding, language,
behavior sequencing); S1, a 200 Hz visuomotor policy translating perception into joint targets; and
S0, a 1 kHz whole-body controller underneath both:

> S0 is a foundation model for human-like whole-body control: a learned prior over how people move
> while maintaining balance and stability. It is the backbone of physical embodiment for Helix 02:
> while higher layers reason about tasks and plans, S0 ensures every motion is executed smoothly,
> safely, and stably.

S2 and S1 together read as a vision-language-action stack. S0 is this series, scaled: a learned
whole-body control prior underneath it, trained separately but running in lock-step at inference
time. My guess is that this becomes the standard shape: a fast, fixed whole-body control layer
with a slower reasoning stack on top, not one policy trained end to end to do everything. This
project is a small, concrete version of that same layering — a style prior and a task reward,
each pulling weight the other can't.
