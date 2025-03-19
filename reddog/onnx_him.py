import os
from dataclasses import dataclass
from operator import add, mul
from threading import Event
from typing import List

import numpy as np
import onnxruntime as ort

import time

import mujoco.viewer
import mujoco
import yaml
import matplotlib.pyplot as plt
import argparse

NUM_MOTOR = 12

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    return np.array([
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz)
    ], dtype=np.float32)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = config["xml_path"]
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        one_step_obs_size = config["one_step_obs_size"]
        obs_buffer_size = config["obs_buffer_size"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)
    
    target_dof_pos = default_angles.copy()
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Record data
    lin_vel_data_list = []
    ang_vel_data_list = []
    gravity_b_list = []
    joint_vel_list = []
    action_list = []
    
    # Load ONNX model
    policy_file_name = '"pre_train/him/KP10/v2/policy.onnx'
    inference_session = ort.InferenceSession(policy_file_name)

    obs_buffer = np.zeros((1, one_step_obs_size * obs_buffer_size), dtype=np.float32)
    counter = 0
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            tau = pd_control(target_dof_pos, d.sensordata[:NUM_MOTOR], kps, np.zeros(12, dtype=np.float32), d.sensordata[NUM_MOTOR:NUM_MOTOR + NUM_MOTOR], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            
            counter += 1
            if counter % control_decimation == 0 and counter > 0:
                qpos = d.sensordata[:12].astype(np.float32)
                qvel = d.sensordata[12:24].astype(np.float32)
                ang_vel_I = d.sensordata[52:55].astype(np.float32)
                gravity_b = get_gravity_orientation(d.sensordata[36:40])
                cmd_vel = np.array(config["cmd_init"], dtype=np.float32)

                obs_list = [
                    cmd_vel * cmd_scale,
                    ang_vel_I * ang_vel_scale,
                    gravity_b,
                    (qpos - default_angles) * dof_pos_scale,
                    qvel * dof_vel_scale,
                    action.astype(np.float32)
                ]
                ## Record Data ##
                ang_vel_data_list.append(ang_vel_I * ang_vel_scale)
                gravity_b_list.append(gravity_b)
                joint_vel_list.append(qvel * dof_vel_scale)
                action_list.append(action)

                obs_array = np.concatenate(obs_list, axis=0).astype(np.float32).reshape(1, -1)
                obs_array = np.clip(obs_array, -100, 100)
                
                obs_buffer = np.concatenate([obs_array, obs_buffer[:, :-one_step_obs_size]], axis=1)
                
                # Policy inference
                action = inference_session.run(None, {inference_session.get_inputs()[0].name: obs_buffer})[0].astype(np.float32).squeeze()
                
                target_dof_pos = default_angles if counter < 300 else action * action_scale + default_angles
            
            viewer.sync()
            
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Plot the collected data after the simulation ends
    plt.figure(figsize=(14, 16))

    plt.subplot(3, 2, 1)
    for i in range(3): 
        plt.plot([step[i] for step in lin_vel_data_list], label=f"Linear Velocity {i}")
    plt.title(f"History Linear Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(3, 2, 2)
    for i in range(3):
        plt.plot([step[i] for step in ang_vel_data_list], label=f"Angular Velocity {i}")
    plt.title(f"History Angular Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(3, 2, 3)
    for i in range(3):
        plt.plot([step[i] for step in gravity_b_list], label=f"Project Gravity {i}")
    plt.title(f"History Project Gravity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(3, 2, 5)
    for i in range(2):
        plt.plot([step[i] for step in joint_vel_list], label=f"Joint Velocity {i}")
    plt.title(f"History Joint Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(3, 2, 6)
    for i in range(2):
        plt.plot([step[i] for step in action_list], label=f"velocity Command {i}")
    plt.title(f"History Torque Command", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.tight_layout()
    plt.show()

