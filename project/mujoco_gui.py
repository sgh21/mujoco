#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/2/29 17:00
# @Author       : Wang Song
# @File         : mujoco_gui.py
# @Software     : PyCharm
# @Description  :
import time
import numpy as np
import mujoco 
import mujoco.viewer
# m=mujoco.MjModel.from_xml_path('./mujoco-py/xmls/tosser.xml')

# m = mujoco.MjModel.from_xml_path('./my_mujoco_platform/UR5+gripper/UR5gripper_2_finger.xml')
m = mujoco.MjModel.from_xml_path('./my_mujoco_platform/UR5e+robotiq85/scene.xml')

# m = mujoco.MjModel.from_xml_path('./my_mujoco_platform/UR5e+robotiq85/scene_doublerobot_vslot.xml')
# m = mujoco.MjModel.from_xml_path('./belt.xml')
d = mujoco.MjData(m)
# m.opt.gravity = (0, 0, 0)  # Turn off gravity.
# d.ctrl[0] = .1  # Apply a control signal to the first actuator.
with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 60*10:
    step_start = time.time()
    print('Total number of DoFs in the model:', m.nv)

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()   # viewer.render() 的更新版本

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)