#!/usr/bin/env python3

# Author: Sherrywu_amour

from collections import defaultdict
import os
from pathlib import Path
import mujoco as mp
import mujoco.viewer
import time
import numpy as np
from simple_pid import PID
from termcolor import colored
from pyquaternion import Quaternion
import cv2 as cv
import matplotlib.pyplot as plt
import copy
import transforms3d
from ikSolver_UR import ikSolver
import pybullet as p
import pybullet_data

ROBOT_URDF_PATH = "./urdf_data/ur5e_hande.urdf"
PLANE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "plane.urdf")

class MJ_Controller(object):
    """
    Class for control of double robotic arm in MuJoCo.
    It can be used on its own, in which case a new model, simulation and viewer will be created.
    It can also be passed these objects when creating an instance, in which case the class can be used
    to perform tasks on an already instantiated simulation.
    """

    def __init__(self, model=None, viewer=None):
        if model is None:
            self.model = mp.MjModel.from_xml_path("./UR5e+robotiq85/scene_doublerobot_vslot.xml")
        else:
            self.model = model
        self.data = mp.MjData(self.model)  #
        # self.sim = mp.MjSim(self.model) if simulation is None else simulation #
        self.viewer = mp.viewer.launch_passive(self.model, self.data) if viewer is None else viewer
        # self.viewer = mp.MjViewer(self.sim) if viewer is None else viewer   # 废弃版本
        self.create_lists()
        self.groups = defaultdict(list)
        self.groups["All"] = list(range(len(self.data.ctrl)))
        self.create_group("Arm", [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
        self.create_group("Arm0", [0, 1, 2, 3, 4, 5])
        self.create_group("Arm1", [7, 8, 9, 10, 11, 12])
        self.create_group("Gripper0", [6])
        self.create_group("Gripper1", [13])
        self.create_group("Grippers", [6, 13])
        self.actuated_joint_ids = np.array([i[2] for i in self.actuators])
        self.reached_target = False
        self.current_output = np.zeros(len(self.data.ctrl))
        self.image_counter = 0
        self.cam_matrix = None
        self.cam_init = False
        self.last_movement_steps = 0
        # cable的起点与终点位姿
        self.cable_startposition = self.data.body('B_first').xpos
        self.cable_startorientation_wxyz = self.data.body('B_first').xquat
        self.cable_endposition = self.data.body('B_last').xpos
        self.cable_endorientation_wxyz = self.data.body('B_last').xquat
        # 一些常用的变换矩阵
        self.rotyn = transforms3d.euler.euler2mat(0, np.radians(-90), 0, 'sxyz')
        self.rotxp = transforms3d.euler.euler2mat(np.radians(90), 0, 0, 'sxyz')
        self.rotzp = transforms3d.euler.euler2mat(0, 0, np.radians(90), 'sxyz')
        self.rotzn = transforms3d.euler.euler2mat(0, 0, np.radians(-90), 'sxyz')
        # 机器人0基座相对于世界坐标系的变换矩阵，形状为(4, 4)
        self.base0_position = self.data.body('robot0:rethink:pedestal').xpos
        self.T0 = np.array([[-1, 0, 0, self.base0_position[0]],
                           [0, -1, 0, self.base0_position[1]],
                           [0, 0, 1, self.base0_position[2]],
                           [0, 0, 0, 1]])
        # 机器人1基座相对于世界坐标系的变换矩阵，形状为(4, 4)
        self.base1_position = self.data.body('robot1:rethink:pedestal').xpos
        self.T1 = np.array([[-1, 0, 0, self.base1_position[0]],
                            [0, -1, 0, self.base1_position[1]],
                            [0, 0, 1, self.base1_position[2]],
                            [0, 0, 0, 1]])
        # inverse kinematics
        ur_a = np.array([0, -0.425, -0.3922, 0, 0, 0])
        ur_d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
        ur_alpha = np.array([0, np.deg2rad(90), 0, 0, np.deg2rad(90), np.deg2rad(-90)])
        self.ik = ikSolver(ur_a, ur_d, ur_alpha)
        # # self.move_group_to_joint_target()

        # pybullet setting
        p.connect(p.DIRECT)
        self.ur5e_pybullet = self.load_robot('ur5e', [0, 0, 0])  #
        self.ur5e_forattainjudge = self.load_robot('ur5e1', [0, 0, 0])  #

    # ------------------------pybullet for iksolver--------------------------------

    def load_robot(self, robot_name, robot_position):
        flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS #p.URDF_USE_SELF_COLLISION
        robot_name = p.loadURDF(ROBOT_URDF_PATH, robot_position, [0, 0, 0, 1], flags=flags, useFixedBase=True)
        # robot_name = p.loadURDF(ROBOT_URDF_PATH, robot_position, [0, 0, 0, 1], useFixedBase=True)
        return robot_name

    def calculate_ik0(self, position, orientation_wxyz, rest_poses=None):
        if rest_poses is None:
            rest_poses = [0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]

        # wxyz trans to xyzw
        quaternion = [orientation_wxyz[1], orientation_wxyz[2], orientation_wxyz[3], orientation_wxyz[0]]
        position = [position[0] - self.base0_position[0], position[1] - self.base0_position[1], position[2] - self.base0_position[2]]

        lower_limits = [-np.pi] * 6
        upper_limits = [np.pi] * 6
        joint_ranges = [2 * np.pi] * 6

        joint_angles_arm = p.calculateInverseKinematics(
            self.ur5e_pybullet, 7, position, quaternion, # self.end_effector_index 7
            jointDamping=[0.01] * 15, upperLimits=upper_limits,
            lowerLimits=lower_limits, jointRanges=joint_ranges,
            restPoses=rest_poses
        )
        joint_angles_arm = list(joint_angles_arm[0:6])
        return joint_angles_arm

    def calculate_ik1(self, position, orientation_wxyz, rest_poses=None):
        if rest_poses is None:
            rest_poses = [0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]

        # wxyz trans to xyzw
        quaternion = [orientation_wxyz[1], orientation_wxyz[2], orientation_wxyz[3], orientation_wxyz[0]]
        position = [position[0] - self.base1_position[0], position[1] - self.base1_position[1], position[2] - self.base1_position[2]]

        lower_limits = [-np.pi] * 6
        upper_limits = [np.pi] * 6
        joint_ranges = [2 * np.pi] * 6

        joint_angles_arm = p.calculateInverseKinematics(
            self.ur5e_pybullet, 7, position, quaternion, # self.end_effector_index 7
            jointDamping=[0.01] * 15, upperLimits=upper_limits,
            lowerLimits=lower_limits, jointRanges=joint_ranges,
            restPoses=rest_poses
        )
        joint_angles_arm = list(joint_angles_arm[0:6])
        return joint_angles_arm

    def set_joint_angles_arm(self, joint_angles):
        # 这里用不同的机械臂是为防止set后的构型使得逆解出现突变2pi的问题
        for i in range(6):
            p.resetJointState(self.ur5e_pybullet, i + 2, joint_angles[i])  # ur5e_forattainjudge

    def get_current_pose(self):
        # 这里用不同的机械臂是为防止set后的构型使得逆解出现突变2pi的问题
        linkstate_arm = p.getLinkState(self.ur5e_pybullet, 7, computeForwardKinematics=True) #
        position_arm, orientation_arm = linkstate_arm[4], linkstate_arm[5] # the URDF link postition and rotation
        return position_arm, orientation_arm

    def disconnect_pybullet(self):
        p.disconnect()

    # -----------------------end for pybullet--------------------------------------

    # -----------------------read and write files----------------------------------
    def read_matrix_from_file(self, filename):
        # 打开文件以读取
        with open(filename, 'r') as file:
            # 逐行读取文件内容
            lines = file.readlines()

            # 初始化一个空矩阵
            matrix = []

            # 逐行解析文件内容并添加到矩阵中
            for line in lines:
                row = [float(num) for num in line.split()]
                matrix.append(row)

        return matrix

    def append_matrix_to_file(self, matrix, filename):
        # 打开文件以追加
        with open(filename, 'a') as file:
            # 如果matrix是一维的，将其转换为二维
            if np.ndim(matrix) == 1:
                matrix = np.reshape(matrix, (1, -1))
            # 遍历矩阵的每一行
            for row in matrix:
                # 将每个元素转换为字符串并用空格分隔
                row_str = ' '.join(str(e) for e in row)
                # 将行写入文件并添加换行符
                file.write(row_str + '\n')

    # -----------------------end for file operations-------------------------------

    def obtain_cable_info(self):
        """
        获取cable的各点位姿信息
        """
        listpos = []
        idstart = self.model.body('B_first').id
        idend = self.model.body('B_last').id
        startpos = self.data.body('B_first').xpos
        for i in range(idstart, idend+1):
            body_id = self.model.body(i).name
            body_position = self.data.body(i).xpos
            body_quat = self.data.body(i).xquat
            if i < idend+1: # idend-1
                listpos = listpos + list(body_position-startpos)
            print(f"Body ID: {body_id}, Body Position: {body_position}, Body quat: {body_quat}")
        print(f"listpos: {listpos}")
        return listpos

    def create_group(self, group_name, idx_list):
        """
        Allows the user to create custom objects for controlling groups of joints.
        The method show_model_info can be used to get lists of joints and actuators.

        Args:
            group_name: String defining the désired name of the group.
            idx_list: List containing the IDs of the actuators that will belong to this group.
        """

        try:
            assert len(idx_list) <= len(self.data.ctrl), "Too many joints specified!"
            assert (
                group_name not in self.groups.keys()
            ), "A group with name {} already exists!".format(group_name)
            assert np.max(idx_list) <= len(
                self.data.ctrl
            ), "List contains invalid actuator ID (too high)"

            self.groups[group_name] = idx_list
            print("Created new control group '{}'.".format(group_name))

        except Exception as e:
            print(e)
            print("Could not create a new group.")

    def show_model_info(self):
        """
        Displays relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges.
        Also gives info on which actuators control which joints and which joints are included in the kinematic chain,
        as well as the PID controller info for each actuator.
        """

        print("\nNumber of bodies: {}".format(self.model.nbody))
        for i in range(self.model.nbody):
            print("Body ID: {}, Body Name: {}".format(i, self.model.body(i).name))

        print("\nNumber of joints: {}".format(self.model.njnt))
        for i in range(self.model.njnt):
            print(
                "Joint ID: {}, Joint Name: {}, Limits: {}".format(
                    i, self.model.joint(i).name, self.model.jnt_range[i]
                )
            )

        print("\nNumber of Actuators: {}".format(len(self.data.ctrl)))
        for i in range(len(self.data.ctrl)):
            print(
                "Actuator ID: {}, Actuator Name: {}, Controlled Joint ID: {}, Control Range: {}".format(
                    i,
                    self.model.actuator(i).name,
                    self.actuators[i][2],
                    self.model.actuator_ctrlrange[i],
                )
            )

        # print("\nJoints in kinematic chain: {}".format([i.name for i in self.ee_chain.links]))

        print("\nPID Info: \n")
        for i in range(len(self.actuators)):
            print(
                "{}: P: {}, I: {}, D: {}, setpoint: {}, sample_time: {}".format(
                    self.actuators[i][3],
                    self.actuators[i][4].tunings[0],
                    self.actuators[i][4].tunings[1],
                    self.actuators[i][4].tunings[2],
                    self.actuators[i][4].setpoint,
                    self.actuators[i][4].sample_time,
                )
            )

        print("\n Camera Info: \n")
        for i in range(self.model.ncam):
            print(
                "Camera ID: {}, Camera Name: {}, Camera FOV (y, degrees): {}, Position: {}, Orientation: {}".format(
                    i,
                    self.model.cam_bodyid[i],
                    self.model.cam_fovy[i],
                    self.model.cam_pos0[i],
                    self.model.cam_mat0[i],
                )
            )

    def create_lists(self):
        """
        Creates some basic lists and fill them with initial values. This function is called in the class costructor.
        The following lists/dictionaries are created:

        - controller_list: Contains a controller for each of the actuated joints. This is done so that different gains may be
        specified for each controller.

        - current_joint_value_targets: Same as the current setpoints for all controllers, created for convenience.

        - current_output = A list containing the ouput values of all the controllers. This list is only initiated here, its
        values are overwritten at the first simulation step.

        - actuators: 2D list, each entry represents one actuator and contains:
            0 actuator ID
            1 actuator name
            2 joint ID of the joint controlled by this actuator
            3 joint name
            4 controller for controlling the actuator
        """

        self.controller_list = []

        # Values for training
        sample_time = 0.0001
        # p_scale = 1
        p_scale = 3
        i_scale = 0.0
        i_gripper = 0
        d_scale = 0.1
        # ----------------------for robot0-------------------------
        self.controller_list.append(
            PID(
                7 * p_scale, 0.0 * i_scale, 1.1 * d_scale, # 7
                setpoint=-1.570796, output_limits=(-2, 2), sample_time=sample_time,
            )
        )  # Shoulder Pan Joint
        self.controller_list.append(
            PID(
                20 * p_scale,   # 20
                0.0 * i_scale,
                80.0 * d_scale, # 1.0
                setpoint=-1.570796, output_limits=(-2, 2), sample_time=sample_time,
            )
        )  # Shoulder Lift Joint
        self.controller_list.append(
            PID(
                5 * p_scale, 0.0 * i_scale, 0.5 * d_scale,
                setpoint=1.570796, output_limits=(-2, 2), sample_time=sample_time,
            )
        )  # Elbow Joint
        self.controller_list.append(
            PID(
                7 * p_scale, 0.0 * i_scale, 0.1 * d_scale,
                setpoint=0,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Wrist 1 Joint
        self.controller_list.append(
            PID(
                5 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=0,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Wrist 2 Joint
        self.controller_list.append(
            PID(
                5 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=0.0,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Wrist 3 Joint
        self.controller_list.append(
            PID(
                2.5 * p_scale,  # 2.5
                i_gripper,
                0.00 * d_scale,
                setpoint=0.0,
                output_limits=(-1, 1),  # 1
                sample_time=sample_time,
            )
        )  # Gripper Joint
        # ----------------------for robot1-------------------------
        self.controller_list.append(
            PID(
                7 * p_scale, 0.0 * i_scale, 1.1 * d_scale,
                setpoint=0, output_limits=(-2, 2), sample_time=sample_time,
            )
        )  # Shoulder Pan Joint
        self.controller_list.append(
            PID(
                20 * p_scale,  # 20
                0.0 * i_scale,
                80.0 * d_scale, # 1
                setpoint=-1.57, output_limits=(-2, 2), sample_time=sample_time,
            )
        )  # Shoulder Lift Joint
        self.controller_list.append(
            PID(
                5 * p_scale, 0.0 * i_scale, 0.5 * d_scale,
                setpoint=1.57, output_limits=(-2, 2), sample_time=sample_time,
            )
        )  # Elbow Joint
        self.controller_list.append(
            PID(
                7 * p_scale, 0.0 * i_scale, 0.1 * d_scale,
                setpoint=-1.57,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Wrist 1 Joint
        self.controller_list.append(
            PID(
                5 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=-1.57,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Wrist 2 Joint
        self.controller_list.append(
            PID(
                5 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=0.0,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Wrist 3 Joint
        self.controller_list.append(
            PID(
                2.5 * p_scale,  # 2.5
                i_gripper,
                0.00 * d_scale,
                setpoint=0.0,
                output_limits=(-1, 1),  # 1
                sample_time=sample_time,
            )
        )  # Gripper Joint
        # self.controller_list.append(PID(10.5*p_scale, 0.2, 0.1*d_scale, setpoint=0.0, output_limits=(-1, 1), sample_time=sample_time)) # Gripper Joint
        # self.controller_list.append(PID(2*p_scale, 0.1*i_scale, 0.05*d_scale, setpoint=0.2, output_limits=(-0.5, 0.8), sample_time=sample_time)) # Finger 2 Joint 1
        # self.controller_list.append(PID(1*p_scale, 0.1*i_scale, 0.05*d_scale, setpoint=0.0, output_limits=(-0.5, 0.8), sample_time=sample_time)) # Middle Finger Joint 1
        # self.controller_list.append(PID(1*p_scale, 0.1*i_scale, 0.05*d_scale, setpoint=-0.1, output_limits=(-0.8, 0.8), sample_time=sample_time)) # Gripperpalm Finger 1 Joint

        self.current_target_joint_values = [
            self.controller_list[i].setpoint for i in range(len(self.data.ctrl))
        ]

        self.current_target_joint_values = np.array(self.current_target_joint_values)

        self.current_output = [controller(0) for controller in self.controller_list]
        self.actuators = []
        for i in range(len(self.data.ctrl)):
            item = [i, self.model.actuator(i).name]  # [i, self.model.actuator_id2name(i)]
            item.append(self.model.actuator_trnid[i][0])
            item.append(self.model.joint(self.model.actuator_trnid[i][0]).name)
            item.append(self.controller_list[i])
            self.actuators.append(item)

    def get_current_acutator_values(self, group="All"):
        """
        Returns the current joint values of the actuators.
        """
        ids = self.groups[group]
        values = np.zeros(len(ids))
        current_joint_values = self.data.qpos[self.model.jnt_qposadr[self.actuated_joint_ids]]
        for i, v in enumerate(ids):
            values[i] = current_joint_values[v]
        return values

    def move_group_to_joint_target(
        self,
        group="All",
        target=None,
        tolerance=0.1,
        max_steps=10000,
        plot=False,
        marker=False,
        render=True,
        quiet=False,
    ):
        """
        Moves the specified joint group to a joint target.

        Args:
            group: String specifying the group to move.
            target: List of target joint values for the group.
            tolerance: Threshold within which the error of each joint must be before the method finishes.
            max_steps: maximum number of steps to actuate before breaking
            plot: If True, a .png image of the group joint trajectories will be saved to the local directory.
                  This can be used for PID tuning in case of overshoot etc. The name of the file will be "Joint_angles_" + a number.
            marker: If True, a colored visual marker will be added into the scene to visualize the current
                    cartesian target.
        """

        try:
            assert group in self.groups.keys(), "No group with name {} exists!".format(group)
            if target is not None:
                assert len(target) == len(
                    self.groups[group]
                ), "Mismatching target dimensions for group {}!".format(group)
            ids = self.groups[group]
            steps = 1
            result = ""
            if plot:
                self.plot_list = defaultdict(list)
            self.reached_target = False
            deltas = np.zeros(len(self.data.ctrl))

            if target is not None:
                for i, v in enumerate(ids):
                    self.current_target_joint_values[v] = target[i]
                    # print('Target joint value: {}: {}'.format(v, self.current_target_joint_values[v]))

            for j in range(len(self.data.ctrl)):
                # Update the setpoints of the relevant controllers for the group
                self.actuators[j][4].setpoint = self.current_target_joint_values[j]
                # print('Setpoint {}: {}'.format(j, self.actuators[j][4].setpoint))

            while not self.reached_target:
                # 若有绳子存在，得注意qpos序号与关节序号不同
                current_joint_values = self.data.qpos[self.model.jnt_qposadr[self.actuated_joint_ids]]

                # self.get_image_data(width=200, height=200, show=True)

                # We still want to actuate all motors towards their targets, otherwise the joints of non-controlled
                # groups will start to drift
                for j in range(len(self.data.ctrl)):
                    self.current_output[j] = self.actuators[j][4](current_joint_values[j])
                    self.data.ctrl[j] = self.current_output[j]
                # self.data.ctrl[0:6] = [-1.8 ,-0.8, 1.2 ,-0.4, -0.93, 0]  # target

                for i in ids:
                    deltas[i] = abs(self.current_target_joint_values[i] - current_joint_values[i])

                if steps % 1000 == 0 and target is not None and not quiet:
                    print(
                        "Moving group {} to joint target! Max. delta: {}, Joint: {}".format(
                            group, max(deltas), self.actuators[np.argmax(deltas)][2]
                        )
                    )

                if plot and steps % 2 == 0:
                    self.fill_plot_list(group, steps)

                if marker:
                    print('no function named add_marker')
                    # self.add_marker(self.current_carthesian_target)
                    # self.add_marker(temp)

                if max(deltas) < tolerance:
                    if target is not None and not quiet:
                        print(
                            colored(
                                "Joint values for group {} within requested tolerance! ({} steps)".format(
                                    group, steps
                                ),
                                color="green",
                                attrs=["bold"],
                            )
                        )
                    result = "success"
                    self.reached_target = True
                    # break

                if steps > max_steps:
                    if not quiet:
                        print(
                            colored(
                                "Max number of steps reached: {}".format(max_steps),
                                color="red",
                                attrs=["bold"],
                            )
                        )
                        print("Deltas: ", deltas)
                        print("max Deltas: ", max(deltas))
                    result = "max. steps reached: {}".format(max_steps)
                    break

                # self.sim.step() 废弃版本
                mp.mj_step(self.model, self.data)

                if render:
                    self.viewer.sync()
                steps += 1

            self.last_movement_steps = steps

            if plot:
                self.create_joint_angle_plot(group=group, tolerance=tolerance)

            return result

        except Exception as e:
            print(e)
            print("Could not move to requested joint target.")


    def open_gripper(self, half=False, **kwargs):
        """
        Opens the gripper while keeping the arm in a steady position.
        """
        # print('Open: ', self.sim.data.qpos[self.actuated_joint_ids][self.groups['Gripper']])
        return (
            self.move_group_to_joint_target(
                group="Grippers", target=[0.0, 0.0], max_steps=1000, tolerance=0.03, **kwargs
            )
            if half
            else
            self.move_group_to_joint_target(
                group="Grippers", target=[0.3, 0.3], max_steps=1000, tolerance=0.03, **kwargs
            )
        )

    def close_gripper0(self, **kwargs):
        # def close_gripper(self, render=True, max_steps=1000, plot=False, quiet=True):
        """
        Closes the gripper while keeping the arm in a steady position.
        """
        # result = self.move_group_to_joint_target(group='Gripper', target=[-0.4], tolerance=0.05, **kwargs)
        # print('Closed: ', self.sim.data.qpos[self.actuated_joint_ids][self.groups['Gripper']])
        # result = self.move_group_to_joint_target(group='Gripper', target=[0.45, 0.45, 0.55, -0.17], tolerance=0.05, max_steps=max_steps, render=render, marker=True, quiet=quiet, plot=plot)
        return self.move_group_to_joint_target(
            group="Gripper0", target=[0.7], tolerance=0.01, **kwargs
        )

    def close_gripper1(self, **kwargs):
        # def close_gripper(self, render=True, max_steps=1000, plot=False, quiet=True):
        """
        Closes the gripper while keeping the arm in a steady position.
        """
        return self.move_group_to_joint_target(
            group="Gripper1", target=[0.78], tolerance=0.01, **kwargs
        )

    def grasp(self, **kwargs):
        # def grasp(self, render=True, plot=False):
        """
        Attempts a grasp at the current location and prints some feedback on weather it was successful
        """
        result0 = self.close_gripper0(max_steps=300, **kwargs)
        result = self.close_gripper1(max_steps=300, **kwargs)

        return result != "success"

    def AdjustRobotsIfNeeded(self, worldee0_pos, worldee0_quat_wxyz, worldee1_pos, worldee1_quat_wxyz, q_last):
        """
        判断机器人能否达到目标位置，若不能，则调整机器人的位置，同时保持线的相对末端位姿不变
        """
        [flag0, delta0] = self.judgeAttainable0(worldee0_pos, worldee0_quat_wxyz, q_last)
        [flag1, delta1] = self.judgeAttainable1(worldee1_pos, worldee1_quat_wxyz, q_last)
        if flag0 and flag1:
            return True, worldee0_pos, worldee0_quat_wxyz, worldee1_pos, worldee1_quat_wxyz
        else:
            print("需要调整机器人位置")
            if not flag0 and not flag1:
                if np.linalg.norm(delta0) > np.linalg.norm(delta1):
                    [success, adee0_pos, adee1_pos] = self.AdjustRobots(worldee0_pos, worldee0_quat_wxyz, worldee1_pos,
                                            worldee1_quat_wxyz, q_last, delta0, mainrobot=0)
                else:
                    [success, adee0_pos, adee1_pos] = self.AdjustRobots(worldee0_pos, worldee0_quat_wxyz, worldee1_pos,
                                            worldee1_quat_wxyz, q_last, delta1, mainrobot=1)
            else:
                if not flag0:
                    [success, adee0_pos, adee1_pos] = self.AdjustRobots(worldee0_pos, worldee0_quat_wxyz, worldee1_pos,
                                            worldee1_quat_wxyz, q_last, delta0, mainrobot=0)
                else:
                    [success, adee0_pos, adee1_pos] = self.AdjustRobots(worldee0_pos, worldee0_quat_wxyz, worldee1_pos,
                                            worldee1_quat_wxyz, q_last, delta1, mainrobot=1)
        if success:
            return True, adee0_pos, worldee0_quat_wxyz, adee1_pos, worldee1_quat_wxyz
        else:
            return False, worldee0_pos, worldee0_quat_wxyz, worldee1_pos, worldee1_quat_wxyz

    def AdjustRobots(self, worldee0_pos, worldee0_quat_wxyz, worldee1_pos, worldee1_quat_wxyz, q_last, delta, mainrobot=0):
        print("mainrobot: ", mainrobot)
        if mainrobot == 0:
            desire_position = [worldee0_pos[0] - self.base0_position[0],
                               worldee0_pos[1] - self.base0_position[1],
                               worldee0_pos[2] - self.base0_position[2]]
        else:
            desire_position = [worldee1_pos[0] - self.base1_position[0],
                               worldee1_pos[1] - self.base1_position[1],
                               worldee1_pos[2] - self.base1_position[2]]
        dir_origin = -np.array(desire_position) / np.linalg.norm(desire_position)
        dir_delta = delta / np.linalg.norm(delta)

        print(np.linalg.norm(delta))
        if np.linalg.norm(delta) > 0.01:
            ad_list = [list(1.2 * np.linalg.norm(delta) * dir_origin), list(1.2 * np.linalg.norm(delta) * dir_delta),
                        list(1.5 * np.linalg.norm(delta) * dir_origin), list(1.5 * np.linalg.norm(delta) * dir_delta),
                        list(1.8 * np.linalg.norm(delta) * dir_origin), list(1.8 * np.linalg.norm(delta) * dir_delta)]
        else:  # 由于角度超限导致需要调整
            ad_list_ = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                       [1/1.72, 1/1.72, 1/1.72], [-1/1.72, 1/1.72, 1/1.72], [1/1.72, -1/1.72, 1/1.72], [-1/1.72, -1/1.72, 1/1.72],
                       [1/1.72, 1/1.72, -1/1.72], [-1/1.72, 1/1.72, -1/1.72], [1/1.72, -1/1.72, -1/1.72], [-1/1.72, -1/1.72, -1/1.72]]
            ad_list = list(0.1 * np.array(ad_list_)) + list(0.2 * np.array(ad_list_)) + list(0.3 * np.array(ad_list_))

        for i in range(len(ad_list)):
            ad_ee0pos = [worldee0_pos[0] + ad_list[i][0], worldee0_pos[1] + ad_list[i][1], worldee0_pos[2] + ad_list[i][2]]
            ad_ee1pos = [worldee1_pos[0] + ad_list[i][0], worldee1_pos[1] + ad_list[i][1], worldee1_pos[2] + ad_list[i][2]]
            [flag0, delta0] = self.judgeAttainable0(ad_ee0pos, worldee0_quat_wxyz, q_last)
            [flag1, delta1] = self.judgeAttainable1(ad_ee1pos, worldee1_quat_wxyz, q_last)
            if flag0 and flag1:
                print("list: ", i)
                return True, ad_ee0pos, ad_ee1pos
        return False, worldee0_pos, worldee1_pos

    def judgeAttainable0(self, ee0_world_position, ee0_world_quat_wxyz, qall_last):
        """
        判断机器人0是否能达到目标位置
        """
        self.set_joint_angles_arm(qall_last[0:6])
        joint_angles0 = self.calculate_ik0(ee0_world_position, ee0_world_quat_wxyz, qall_last[0:6])
        maxvalue = max(abs(x) for x in joint_angles0)
        self.set_joint_angles_arm(joint_angles0)
        [position_arm, orientation_arm] = self.get_current_pose()
        desire_position = [ee0_world_position[0] - self.base0_position[0], ee0_world_position[1] - self.base0_position[1],
                    ee0_world_position[2] - self.base0_position[2]]
        delta = np.array(position_arm) - np.array(desire_position)
        distance = np.linalg.norm(delta)
        if distance <= 0.01 and maxvalue <= 6.28:
            return True, delta
        else:
            return False, delta

    def judgeAttainable1(self, ee1_world_position, ee1_world_quat_wxyz, qall_last):
        """
        判断机器人1是否能达到目标位置
        """
        self.set_joint_angles_arm(qall_last[6:12])
        joint_angles1 = self.calculate_ik1(ee1_world_position, ee1_world_quat_wxyz, qall_last[6:12])
        maxvalue = max(abs(x) for x in joint_angles1)
        self.set_joint_angles_arm(joint_angles1)
        [position_arm, orientation_arm] = self.get_current_pose()
        desire_position = [ee1_world_position[0] - self.base1_position[0],
                           ee1_world_position[1] - self.base1_position[1],
                           ee1_world_position[2] - self.base1_position[2]]
        delta = np.array(position_arm) - np.array(desire_position)
        distance = np.linalg.norm(delta)
        if distance <= 0.01 and maxvalue <= 6.28:
            return True, delta
        else:
            return False, delta

    def move_ee0(self, ee0_world_position, ee0_world_quat_wxyz, qall_last, **kwargs):
        """
        Moves the robot arm so that the gripper center ends up at the requested XYZ-position,
        with a vertical gripper position.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).
            plot: If True, a .png image of the arm joint trajectories will be saved to the local directory.
                  This can be used for PID tuning in case of overshoot etc. The name of the file will be "Joint_angles_" + a number.
            marker: If True, a colored visual marker will be added into the scene to visualize the current
                    cartesian target.
        """
        # joint_angles = [-1.8, -0.8, 1.2, -0.4, -0.93, -0, 0.7]  # # !!
        # [joint_angles0, qs] = self.inv_k0(ee0_world_position, ee0_world_quat_wxyz, qall_last)
        self.set_joint_angles_arm(qall_last[0:6])
        joint_angles0 = self.calculate_ik0(ee0_world_position, ee0_world_quat_wxyz, qall_last[0:6])
        print("robot0_angles:", list(joint_angles0))
        # print("qs0_angles:", list(qs))
        if joint_angles0 is not None:
            result = self.move_group_to_joint_target(group="Arm0", target=joint_angles0, **kwargs)
            # result = self.move_group_to_joint_target(group='Arm', target=joint_angles, tolerance=0.05, plot=plot, marker=marker, max_steps=max_steps, quiet=quiet, render=render)
        else:
            result = "No valid joint angles received, could not move EE to position."
            self.last_movement_steps = 0
        return result

    def move_ee1(self, ee1_world_position, ee1_world_quat_wxyz, qall_last, **kwargs):
        """
        Moves the robot arm so that the gripper center ends up at the requested XYZ-position,
        with a vertical gripper position.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).
            plot: If True, a .png image of the arm joint trajectories will be saved to the local directory.
                  This can be used for PID tuning in case of overshoot etc. The name of the file will be "Joint_angles_" + a number.
            marker: If True, a colored visual marker will be added into the scene to visualize the current
                    cartesian target.
        """
        # joint_angles = [-1.8, -0.8, 1.2, -0.4, -0.93, -0, 0.7]  # # !!
        # [joint_angles1, qs] = self.inv_k1(ee1_world_position, ee1_world_quat_wxyz, qall_last)
        self.set_joint_angles_arm(qall_last[6:12])
        joint_angles1 = self.calculate_ik1(ee1_world_position, ee1_world_quat_wxyz, qall_last[6:12])
        print("robot1_angles:", list(joint_angles1))
        # print("qs1_angles:", list(qs))
        if joint_angles1 is not None:
            result = self.move_group_to_joint_target(group="Arm1", target=joint_angles1, **kwargs)
            # result = self.move_group_to_joint_target(group='Arm', target=joint_angles, tolerance=0.05, plot=plot, marker=marker, max_steps=max_steps, quiet=quiet, render=render)
        else:
            result = "No valid joint angles received, could not move EE to position."
            self.last_movement_steps = 0
        return result

    def move_ee(self, ee0_world_position, ee0_world_quat_wxyz, ee1_world_position, ee1_world_quat_wxyz, qall_last, **kwargs):
        """
        Moves the robot arm so that the gripper center ends up at the requested XYZ-position,
        with a vertical gripper position.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).
            plot: If True, a .png image of the arm joint trajectories will be saved to the local directory.
                  This can be used for PID tuning in case of overshoot etc. The name of the file will be "Joint_angles_" + a number.
            marker: If True, a colored visual marker will be added into the scene to visualize the current
                    cartesian target.
        """
        # joint_angles = [-1.8, -0.8, 1.2, -0.4, -0.93, -0, 0.7]  # # !!
        # [joint_angles0, qs0] = self.inv_k0(ee0_world_position, ee0_world_quat_wxyz, qall_last)
        # [joint_angles1, qs1] = self.inv_k1(ee1_world_position, ee1_world_quat_wxyz, qall_last)
        self.set_joint_angles_arm(qall_last[0:6])
        joint_angles0 = self.calculate_ik0(ee0_world_position, ee0_world_quat_wxyz, qall_last[0:6])
        self.set_joint_angles_arm(qall_last[6:12])
        joint_angles1 = self.calculate_ik1(ee1_world_position, ee1_world_quat_wxyz, qall_last[6:12])
        print("robot0_angles:", list(joint_angles0))
        print("robot1_angles:", list(joint_angles1))
        print("angles:", list(joint_angles0) + list(joint_angles1))
        if joint_angles0 is not None and joint_angles1 is not None:
            result = self.move_group_to_joint_target(group="Arm", target=list(joint_angles0) + list(joint_angles1), **kwargs)
            # result = self.move_group_to_joint_target(group='Arm', target=joint_angles, tolerance=0.05, plot=plot, marker=marker, max_steps=max_steps, quiet=quiet, render=render)
        else:
            result = "No valid joint angles received, could not move EE to position."
            self.last_movement_steps = 0
        return result

    def inv_k0(self, ee0_world_position, ee0_world_quat_wxyz, qall_last):
        """
        Method for solving simple inverse kinematic problems.
        This was developed for top down graspig, therefore the solution will be one where the gripper is
        vertical. This might need adjustment for other gripper models.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).
            qall_last: List of joint angles(two robots) from the last simulation step.

        Returns:
            joint_angles: List of joint angles that will achieve the desired ee position.
        """

        try:
            assert (
                len(ee0_world_position) == 3
            ), "Invalid EE target! Please specify XYZ-coordinates in a list of length 3."
            q0_last = qall_last[0:6]

            # ------------robot0--------------
            [robot_pos, robot_quat_wxyz] = self.transform_to_robot0_frame_wxyz(ee0_world_position, ee0_world_quat_wxyz)
            orientation = self.quaternion_to_euler_xyz(robot_quat_wxyz)

            # position: 3x1 numpy array of the position, xyz; orientation: 3x1 numpy array of the orientation in euler angles xyz
            Tran = self.ik.create_Transformation_Matrix(np.array(robot_pos), np.array(orientation))

            joint_angles0 = self.ik.solveIK(Tran, q0_last)

            joint_angles = joint_angles0
            return joint_angles

        except Exception as e:
            print(e)
            print("Could not find an inverse kinematics solution.")

    def inv_k1(self, ee1_world_position, ee1_world_quat_wxyz, qall_last):
        """
        Method for solving simple inverse kinematic problems.
        This was developed for top down graspig, therefore the solution will be one where the gripper is
        vertical. This might need adjustment for other gripper models.

        Args:
            ee_position: List of XYZ-coordinates of the end-effector (ee_link for UR5 setup).
            qall_last: List of joint angles(two robots) from the last simulation step.

        Returns:
            joint_angles: List of joint angles that will achieve the desired ee position.
        """

        try:
            assert (
                len(ee1_world_position) == 3
            ), "Invalid EE target! Please specify XYZ-coordinates in a list of length 3."
            q1_last = qall_last[6:12]

            # ------------robot1---------------
            [robot_pos, robot_quat_wxyz] = self.transform_to_robot1_frame_wxyz(ee1_world_position, ee1_world_quat_wxyz)
            orientation = self.quaternion_to_euler_xyz(robot_quat_wxyz)

            # position: 3x1 numpy array of the position, xyz; orientation: 3x1 numpy array of the orientation in euler angles xyz
            Tran = self.ik.create_Transformation_Matrix(np.array(robot_pos), np.array(orientation))

            joint_angles1 = self.ik.solveIK(Tran, q1_last)

            joint_angles = joint_angles1
            return joint_angles

        except Exception as e:
            print(e)
            print("Could not find an inverse kinematics solution.")

    def checkIfCableInHand(self):
        """
        判断末端执行器是否仍然抓着绳子
        """
        cable_end = np.array(self.cable_endposition)
        ee1_site = np.array(self.data.site("robot1:eef_site").xpos)
        # 两者原本的标准距离为0.02
        if np.linalg.norm(cable_end - ee1_site) < 0.03:
            return True
        else:
            return False

    def worldGripper_to_worldee(self, gripper_position, gripper_quat_wxyz, robotid = 0):
        """
        将夹爪中心的位置和姿态(也就是待抓的线上点的位姿)转换为末端执行器的位置和姿态，都在世界坐标系中进行。

        参数:
        - gripper_position: 夹爪中心的位置，形状为(3,)
        - gripper_quat_wxyz: 夹爪中心的姿态（四元数，wxyz格式），形状为(4,)

        返回:
        - ee_position: 末端执行器的位置，形状为(3,)
        - ee_quat_wxyz: 末端执行器的姿态（四元数，wxyz格式），形状为(4,)
        """
        # 将世界坐标系下的四元数转换为旋转矩阵
        R = transforms3d.quaternions.quat2mat(gripper_quat_wxyz)

        if robotid == 0:
            Rtran = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            rot_mat = np.dot(R, Rtran)
        else: # robotid == 1
            # one type of catching
            rot_mat = np.dot(R, self.rotyn)
            # # the other type of catching
            # rot_mat = np.dot(R, self.rotxp)
            # rot_mat = np.dot(rot_mat, self.rotzp)

        worldee_quat_wxyz = transforms3d.quaternions.mat2quat(rot_mat)

        # 求位置
        if robotid == 0:
            dgripper2world = np.dot(rot_mat, np.array([0, 0, 0.16]))
        else:  # robotid == 1
            dgripper2world = np.dot(rot_mat, np.array([0, 0, 0.14]))
        worldee_pos = gripper_position - dgripper2world

        # # 不变姿态的情况
        # worldee_quat_wxyz = gripper_quat_wxyz

        return worldee_pos, worldee_quat_wxyz

    def transform_to_robot0_frame_wxyz(self, world_pos, world_quat_wxyz):
        """
        将世界坐标系下的位置和姿态转换到机器人0的基座坐标系下。

        参数:
        - world_pos: 世界坐标系下的位置，形状为(3,)
        - world_quat_wxyz: 世界坐标系下的姿态（四元数，wxyz格式），形状为(4,)
        - T: 机器人基座相对于世界坐标系的变换矩阵，形状为(4, 4)

        返回:
        - robot_pos: 机器人坐标系下的位置，形状为(3,)
        - robot_quat_wxyz: 机器人坐标系下的姿态（四元数，wxyz格式），形状为(4,)
        """
        T_inv = np.linalg.inv(self.T0)

        # 将世界坐标系下的位置转换为齐次坐标
        world_pos_homogeneous = np.append(world_pos, 1)

        # 应用变换矩阵
        robot_pos_homogeneous = np.dot(T_inv, world_pos_homogeneous)

        # 从齐次坐标中提取转换后的位置
        robot_pos = robot_pos_homogeneous[:3]

        # 提取旋转矩阵部分
        R = T_inv[:3, :3]
        # 将世界坐标系下的四元数转换为旋转矩阵
        world_rot_mat = transforms3d.quaternions.quat2mat(world_quat_wxyz)
        # 应用旋转矩阵
        robot_rot_mat = np.dot(R, world_rot_mat)
        # 将转换后的旋转矩阵转换为四元数（wxyz格式）
        robot_quat_wxyz = transforms3d.quaternions.mat2quat(robot_rot_mat)

        return robot_pos, robot_quat_wxyz

    def transform_to_robot1_frame_wxyz(self, world_pos, world_quat_wxyz):
        """
        将世界坐标系下的位置和姿态转换到机器人1的基座坐标系下。

        参数:
        - world_pos: 世界坐标系下的位置，形状为(3,)
        - world_quat_wxyz: 世界坐标系下的姿态（四元数，wxyz格式），形状为(4,)
        - T: 机器人基座相对于世界坐标系的变换矩阵，形状为(4, 4)

        返回:
        - robot_pos: 机器人坐标系下的位置，形状为(3,)
        - robot_quat_wxyz: 机器人坐标系下的姿态（四元数，wxyz格式），形状为(4,)
        """
        T_inv = np.linalg.inv(self.T1)

        # 将世界坐标系下的位置转换为齐次坐标
        world_pos_homogeneous = np.append(world_pos, 1)

        # 应用变换矩阵
        robot_pos_homogeneous = np.dot(T_inv, world_pos_homogeneous)

        # 从齐次坐标中提取转换后的位置
        robot_pos = robot_pos_homogeneous[:3]

        # 提取旋转矩阵部分
        R = T_inv[:3, :3]
        # 将世界坐标系下的四元数转换为旋转矩阵
        world_rot_mat = transforms3d.quaternions.quat2mat(world_quat_wxyz)
        # 应用旋转矩阵
        robot_rot_mat = np.dot(R, world_rot_mat)
        # 将转换后的旋转矩阵转换为四元数（wxyz格式）
        robot_quat_wxyz = transforms3d.quaternions.mat2quat(robot_rot_mat)

        return robot_pos, robot_quat_wxyz

    def quaternion_to_euler_xyz(self, quat_wxyz):
        """
        将四元数转换为XYZ顺序的欧拉角。

        参数:
        - quat: 四元数，格式为(w, x, y, z)

        返回:
        - euler_xyz: XYZ顺序的欧拉角，单位为弧度
        """
        # 将四元数转换为旋转矩阵
        mat = transforms3d.quaternions.quat2mat(quat_wxyz)
        # 将旋转矩阵转换为欧拉角（XYZ顺序）
        euler_xyz = transforms3d.euler.mat2euler(mat, 'sxyz')
        return euler_xyz

    def euler_to_quaternion_wxyz(self, euler_xyz):
        """
        Convert Euler angles to quaternion.

        Args:
            euler_xyz: Euler angles in XYZ order, in radians.

        Returns:
            quat_wxyz: Quaternion in wxyz order.
        """
        # Convert Euler angles to quaternion (xyzw order)
        quat_wxyz = transforms3d.euler.euler2quat(euler_xyz[0], euler_xyz[1], euler_xyz[2], 'sxyz')
        return quat_wxyz

    def display_current_values(self):
        """
        Debug method, simply displays some relevant data at the time of the call.
        """

        print("\n################################################")
        print("CURRENT JOINT POSITIONS (ACTUATED)")
        print("################################################")
        for i in range(len(self.actuated_joint_ids)):
            print(
                "Current angle for joint {}: {}".format(
                    self.actuators[i][3], self.data.qpos[self.model.jnt_qposadr[self.actuated_joint_ids[i]]]
                )
            )

        print("\n################################################")
        print("CURRENT BODY POSITIONS")
        print("################################################")
        for i in range(self.model.nbody):
            print(
                "Current position for body {}: {}".format(
                    self.model.body(i).name, self.data.xpos[i]
                )
            )

        print("\n################################################")
        print("CURRENT BODY ROTATION MATRIZES")
        print("################################################")
        for i in range(self.model.nbody):
            print(
                "Current rotation for body {}: {}".format(
                    self.model.body(i).name, self.data.xmat[i]
                )
            )

        print("\n################################################")
        print("CURRENT BODY ROTATION QUATERNIONS (w,x,y,z)")
        print("################################################")
        for i in range(self.model.nbody):
            print(
                "Current rotation for body {}: {}".format(
                    self.model.body(i).name, self.data.xquat[i]
                )
            )

        print("\n################################################")
        print("CURRENT ACTUATOR CONTROLS")
        print("################################################")
        for i in range(len(self.data.ctrl)):
            print(
                "Current activation of actuator {}: {}".format(
                    self.actuators[i][1], self.data.ctrl[i]
                )
            )

    def stay(self, duration, render=True):
        """
        Holds the current position by actuating the joints towards their current target position.

        Args:
            duration: Time in ms to hold the position.
        """

        # print('Holding position!')
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:
            self.move_group_to_joint_target(
                max_steps=10, tolerance=0.0000001, plot=False, quiet=True, render=render
            )
            elapsed = (time.time() - starting_time) * 1000
        # print('Moving on...')

    def fill_plot_list(self, group, step):
        """
        Creates a two dimensional list of joint angles for plotting.

        Args:
            group: The group involved in the movement.
            step: The step of the trajectory the values correspond to.
        """

        for i in self.groups[group]:
            self.plot_list[self.actuators[i][3]].append(
                self.data.qpos[self.model.jnt_qposadr[self.actuated_joint_ids]][i]    #
            )
        self.plot_list["Steps"].append(step)

    def create_joint_angle_plot(self, group, tolerance):
        """
        Saves the recorded joint values as a .png-file. The values for each joint of the group are
        put in a seperate subplot.

        Args:
            group: The group the stored values belong to.
            tolerance: The tolerance value that the joints were required to be in.
        """

        self.image_counter += 1
        keys = list(self.plot_list.keys())
        number_subplots = len(self.plot_list) - 1
        columns = 3
        rows = (number_subplots // columns) + (number_subplots % columns)

        position = range(1, number_subplots + 1)
        fig = plt.figure(1, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.05)

        for i in range(number_subplots):
            axis = fig.add_subplot(rows, columns, position[i])
            axis.plot(self.plot_list["Steps"], self.plot_list[keys[i]])
            axis.set_title(keys[i])
            axis.set_xlabel(keys[-1])
            axis.set_ylabel("Joint angle [rad]")
            axis.xaxis.set_label_coords(0.05, -0.1)
            axis.yaxis.set_label_coords(1.05, 0.5)
            axis.axhline(
                self.current_target_joint_values[self.groups[group][i]], color="g", linestyle="--"
            )
            axis.axhline(
                self.current_target_joint_values[self.groups[group][i]] + tolerance,
                color="r",
                linestyle="--",
            )
            axis.axhline(
                self.current_target_joint_values[self.groups[group][i]] - tolerance,
                color="r",
                linestyle="--",
            )

        filename = "Joint_values_{}.png".format(self.image_counter)
        plt.savefig(filename)
        print(
            colored(
                "Saved trajectory to {}.".format(filename),
                color="yellow",
                on_color="on_grey",
                attrs=["bold"],
            )
        )
        plt.clf()

    def get_image_data(self, show=False, camera="top_down", width=200, height=200):
        """
        Returns the RGB and depth images of the provided camera.

        Args:
            show: If True displays the images for five seconds or until a key is pressed.
            camera: String specifying the name of the camera to use.
        """

        rgb, depth = copy.deepcopy(
            self.sim.render(width=width, height=height, camera_name=camera, depth=True)  #
        )
        if show:
            cv.imshow("rbg", cv.cvtColor(rgb, cv.COLOR_BGR2RGB))
            # cv.imshow('depth', depth)
            cv.waitKey(1)
            # cv.waitKey(delay=5000)
            # cv.destroyAllWindows()

        return np.array(np.fliplr(np.flipud(rgb))), np.array(np.fliplr(np.flipud(depth)))

    def depth_2_meters(self, depth):
        """
        Converts the depth array delivered by MuJoCo (values between 0 and 1) into actual m values.

        Args:
            depth: The depth array to be converted.
        """

        extend = self.model.stat.extent
        near = self.model.vis.map.znear * extend
        far = self.model.vis.map.zfar * extend
        return near / (1 - depth * (1 - near / far))

    def create_camera_data(self, width, height, camera):
        """
        Initializes all camera parameters that only need to be calculated once.
        """

        cam_id = self.model.camera_name2id(camera)
        # Get field of view
        fovy = self.model.cam_fovy[cam_id]
        # Calculate focal length
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        # Construct camera matrix
        self.cam_matrix = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        # Rotation of camera in world coordinates
        self.cam_rot_mat = self.model.cam_mat0[cam_id]
        self.cam_rot_mat = np.reshape(self.cam_rot_mat, (3, 3))
        # Position of camera in world coordinates
        self.cam_pos = self.model.cam_pos0[cam_id]
        self.cam_init = True

    def world_2_pixel(self, world_coordinate, width=200, height=200, camera="top_down"):
        """
        Takes a XYZ world position and transforms it into pixel coordinates.
        Mainly implemented for testing the correctness of the camera matrix, focal length etc.

        Args:
            world_coordinate: XYZ world coordinate to be transformed into pixel space.
            width: Width of the image (pixel).
            height: Height of the image (pixel).
            camera: Name of camera used to obtain the image.
        """

        if not self.cam_init:
            self.create_camera_data(width, height, camera)

        # Homogeneous image point
        hom_pixel = self.cam_matrix @ self.cam_rot_mat @ (world_coordinate - self.cam_pos)
        # Real image point
        pixel = hom_pixel[:2] / hom_pixel[2]

        return np.round(pixel[0]).astype(int), np.round(pixel[1]).astype(int)

    def pixel_2_world(self, pixel_x, pixel_y, depth, width=200, height=200, camera="top_down"):
        """
        Converts pixel coordinates into world coordinates.

        Args:
            pixel_x: X-coordinate in pixel space.
            pixel_y: Y-coordinate in pixel space.
            depth: Depth value corresponding to the pixel.
            width: Width of the image (pixel).
            height: Height of the image (pixel).
            camera: Name of camera used to obtain the image.
        """

        if not self.cam_init:
            self.create_camera_data(width, height, camera)

        # Create coordinate vector
        pixel_coord = np.array([pixel_x, pixel_y, 1]) * (-depth)
        # Get position relative to camera
        pos_c = np.linalg.inv(self.cam_matrix) @ pixel_coord
        # Get world position
        pos_w = np.linalg.inv(self.cam_rot_mat) @ (pos_c + self.cam_pos)

        return pos_w

    def set_group_joint_target(self, group, target):

        idx = self.groups[group]
        try:
            assert len(target) == len(
                idx
            ), "Length of the target must match the number of actuated joints in the group."
            self.current_target_joint_values[idx] = target

        except Exception as e:
            print(e)
            print(f"Could not set new group joint target for group {group}")


    @property
    def last_steps(self):
        return self.last_movement_steps
