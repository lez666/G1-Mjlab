"""Generate a stand motion npz from a task's default robot pose."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg


@dataclass
class StandMotionConfig:
  task_id: str = "Mjlab-Tracking-Flat-Unitree-G1"
  output_file: str = "/home/wasabi/Kevin/g1_npz/g1_stand.npz"
  duration_s: float = 2.0
  fps: float = 50.0
  device: str = "cuda:0"


def _repeat_frame(frame: np.ndarray, frames: int) -> np.ndarray:
  return np.repeat(frame[None, ...], frames, axis=0).astype(np.float32)


def main(cfg: StandMotionConfig) -> None:
  if cfg.device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU.")
    cfg.device = "cpu"

  # Populate task registry.
  import mjlab.tasks  # noqa: F401

  if cfg.task_id not in list_tasks():
    raise ValueError(
      f"Unknown task_id '{cfg.task_id}'. Available: {', '.join(list_tasks())}"
    )

  env_cfg = load_env_cfg(cfg.task_id, play=True)

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / cfg.fps

  scene = Scene(env_cfg.scene, device=cfg.device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=cfg.device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  scene.reset()
  sim.forward()
  scene.update(sim.mj_model.opt.timestep)

  robot = scene["robot"]

  joint_pos = robot.data.joint_pos[0].cpu().numpy()
  joint_vel = robot.data.joint_vel[0].cpu().numpy()
  body_pos_w = robot.data.body_link_pos_w[0].cpu().numpy()
  body_quat_w = robot.data.body_link_quat_w[0].cpu().numpy()
  body_lin_vel_w = robot.data.body_link_lin_vel_w[0].cpu().numpy()
  body_ang_vel_w = robot.data.body_link_ang_vel_w[0].cpu().numpy()

  frame_count = max(int(round(cfg.duration_s * cfg.fps)), 1)

  motion = {
    "joint_pos": _repeat_frame(joint_pos, frame_count),
    "joint_vel": _repeat_frame(joint_vel, frame_count),
    "body_pos_w": _repeat_frame(body_pos_w, frame_count),
    "body_quat_w": _repeat_frame(body_quat_w, frame_count),
    "body_lin_vel_w": _repeat_frame(body_lin_vel_w, frame_count),
    "body_ang_vel_w": _repeat_frame(body_ang_vel_w, frame_count),
  }

  output_path = Path(cfg.output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(str(output_path), **motion)
  print(
    f"[INFO]: Stand motion saved: {output_path} "
    f"(frames={frame_count}, fps={cfg.fps})"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
