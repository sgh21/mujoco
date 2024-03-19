from DoubleRobotController import MJ_Controller
import numpy as np
import transforms3d
# from gym_grasper.controller.MujocoController import MJ_Controller

# create controller instance
controller = MJ_Controller()

controller.model.opt.gravity = (0, 0, 0)  # Turn off gravity. !!

# Display robot information
controller.show_model_info()

controller.obtain_cable_info()

print(controller.data.body("B_first").xpos[0])
print(controller.data.body("B_last").xpos[0])
print(controller.data.body("B_last").xpos[0]-controller.data.body("B_first").xpos[0])

# Wait a second
# controller.stay(60000)

# q_last = controller.get_current_acutator_values(group="Arm")
# controller.move_ee0([-0.22, -0.5, 0.84243], [1,0,0,0], q_last, tolerance=0.05, plot=True, marker=False)  # [0.0, -0.6, 0.95]

controller.stay(1000)

# q_last = controller.get_current_acutator_values(group="Arm")
# print('q_last: ', q_last)
# Rcable = transforms3d.quaternions.quat2mat(controller.cable_startorientation_wxyz)
# Ree = transforms3d.quaternions.quat2mat([1, 0, 0, 0])
# Ree2cable = np.dot(Rcable.T, Ree)
# print('cable_startorientation_mat: ', Rcable)
# print('ee0_orientation_mat: ', Ree)
# print('ee2cable_orientation_mat: ', Ree2cable)
# print('cable_startpostition: ', controller.cable_startposition)

controller.open_gripper()


# Move ee1 to the cable end
q_last = controller.get_current_acutator_values(group="Arm")
[worldee_pos, worldee_quat_wxyz] = controller.worldGripper_to_worldee(controller.cable_endposition, controller.cable_endorientation_wxyz, robotid = 1)
print('cable: ', controller.cable_endposition, controller.cable_endorientation_wxyz)
print('ee1: ', worldee_pos, worldee_quat_wxyz)
print(controller.judgeAttainable1(worldee_pos, worldee_quat_wxyz, q_last))
controller.move_ee1(worldee_pos, worldee_quat_wxyz, q_last, tolerance=0.05, plot=True, marker=False)  # [0.0, -0.6, 0.95]

# q_last = controller.get_current_acutator_values(group="Arm")
# [worldee_pos, worldee_quat_wxyz] = controller.worldGripper_to_worldee(controller.cable_startposition, controller.cable_startorientation_wxyz, robotid = 0)
# print('cable: ', controller.cable_startposition, controller.cable_startorientation_wxyz)
# print('ee0: ', worldee_pos, worldee_quat_wxyz)
# controller.move_ee0(worldee_pos, worldee_quat_wxyz, q_last, tolerance=0.05, plot=True, marker=False)  # [0.0, -0.6, 0.95]

# Wait a second
controller.stay(1000)

# Attempt grasp
controller.grasp()

# Wait a second
controller.stay(3000)

controller.model.opt.gravity = (0, 0, -9.81)  # Turn on gravity. !!

# Wait a second
controller.stay(1000)

# 测试读取文件
A = controller.read_matrix_from_file("./output2.txt")
# 获取矩阵的行数
num_rows = len(A)
# 获取矩阵的列数
num_cols = len(A[0]) if A else 0
print(f"矩阵A的大小为：{num_rows}行，{num_cols}列")

for i in range(num_rows):
    # Move ee1 to the cable end
    q_last = controller.get_current_acutator_values(group="Arm")
    # [ 0.99860993 ,-0.02007115 ,-0.01219456 ,-0.04718734]与[ 0.99988343 , 0.01004924, -0.01125878 , 0.00231739]体现了逆解的不稳定
    # [worldee1_pos, worldee1_quat_wxyz] = controller.worldGripper_to_worldee(controller.cable_endposition+[-0.1, 0, 0], controller.cable_startorientation_wxyz, robotid = 1)
    [worldee0_pos, worldee0_quat_wxyz] = controller.worldGripper_to_worldee(controller.cable_startposition + A[i][0:3],
                                                                            A[i][3:7], robotid=0)
    [worldee1_pos, worldee1_quat_wxyz] = controller.worldGripper_to_worldee(controller.cable_startposition + A[i][7:10],
                                                                            A[i][10:14], robotid=1)
    print('cable1: ', controller.cable_startposition + A[i][7:10], A[i][10:14])
    print('ee1: ', worldee1_pos, worldee1_quat_wxyz)
    print('cable0: ', controller.cable_startposition + A[i][0:3], A[i][3:7])
    print('ee0: ', worldee0_pos, worldee0_quat_wxyz)
    # controller.move_ee(worldee0_pos, worldee0_quat_wxyz, worldee1_pos, worldee1_quat_wxyz, q_last, tolerance=0.05, plot=True, marker=False)  # [0.0, -0.6, 0.95]
    [flag, worldee0_pos, worldee0_quat_wxyz, worldee1_pos, worldee1_quat_wxyz] = controller.AdjustRobotsIfNeeded(
        worldee0_pos, worldee0_quat_wxyz, worldee1_pos, worldee1_quat_wxyz, q_last)
    print('flag: ', flag)
    print('ad_ee1: ', worldee1_pos, worldee1_quat_wxyz)
    print('ad_ee0: ', worldee0_pos, worldee0_quat_wxyz)
    # controller.move_ee1(worldee1_pos, worldee1_quat_wxyz, q_last, tolerance=0.05, plot=True, marker=False)  # [0.0, -0.6, 0.95]

    # Move ee0 to the cable end
    # q_last = controller.get_current_acutator_values(group="Arm")
    # [worldee0_pos, worldee0_quat_wxyz] = controller.worldGripper_to_worldee(controller.cable_startposition+[0.2, 0, 0], controller.cable_startorientation_wxyz, robotid = 0)
    controller.move_ee(worldee0_pos, worldee0_quat_wxyz, worldee1_pos, worldee1_quat_wxyz, q_last, tolerance=0.03,
                       plot=True, marker=False)  # [0.0, -0.6, 0.95]
    # controller.move_ee0(worldee0_pos, worldee0_quat_wxyz, q_last, tolerance=0.05, plot=True, marker=False)  # [0.0, -0.6, 0.95]

    if i in range(6,25) or i in range(37,50) or i in range(58,77) or i in range(81,90):
        listpos = controller.obtain_cable_info()
        controller.append_matrix_to_file(listpos, "500_2.txt")

    print('checkIfCableInHand: ', controller.checkIfCableInHand())
    print()


# controller.stay(10000)

controller.obtain_cable_info()

# # Move ee to the cable end
# q_last = controller.get_current_acutator_values(group="Arm")
# [worldee1_pos, worldee1_quat_wxyz] = controller.worldGripper_to_worldee(controller.cable_endposition+[-0.1, 0.1, 0], controller.cable_startorientation_wxyz, robotid = 1)
# [worldee0_pos, worldee0_quat_wxyz] = controller.worldGripper_to_worldee(controller.cable_startposition+[-0.1, 0.1, 0], controller.cable_startorientation_wxyz, robotid = 0)
# controller.move_ee(worldee0_pos, worldee0_quat_wxyz, worldee1_pos, worldee1_quat_wxyz, q_last, tolerance=0.05, plot=True, marker=False)  # [0.0, -0.6, 0.95]

controller.stay(2000)

# q_last = controller.get_current_acutator_values(group="Arm")
# R = transforms3d.quaternions.quat2mat(controller.cable_endorientation_wxyz)
# R1 = transforms3d.euler.euler2mat(np.radians(30), np.radians(0), np.radians(0), 'sxyz')
# rot_mat = np.dot(R, R1)
# quat_wxyz = transforms3d.quaternions.mat2quat(rot_mat)
# [worldee_pos, worldee_quat_wxyz] = controller.worldGripper_to_worldee(controller.cable_endposition,  quat_wxyz, robotid = 1)
# controller.move_ee1(worldee_pos, worldee_quat_wxyz, q_last, tolerance=0.05, plot=True, marker=False)  # [0.0, -0.6, 0.95]


# Wait a second
controller.stay(3000)


controller.disconnect_pybullet()





