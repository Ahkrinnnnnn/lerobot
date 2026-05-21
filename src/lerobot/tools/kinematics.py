import numpy as np
from scipy.spatial.transform import Rotation as R
import copy


def create_so101_dh_params():
    """[a, alpha, d, theta]"""
    return [
        [0.03, np.pi/2, 0.1, 0.0],         # 关节1-2
        [0.1158, 0.0, 0.0, 1.3316],        # 关节2-3
        [0.135, 0.0, 0.0, -1.3316],        # 关节3-4
        [0.0, np.pi/2, 0.0, np.pi/2] ,     # 关节4-5
        [0.03, 0.0, 0.12, 0.0],            # 关节5-end        
    ]

def create_tool_transform():
    """so101end_T_crpend"""
    # return np.array([
    #     [1.0, 0.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.0],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])

    return np.array([  
        [-0.81022953, -0.5713876,   0.1305539,  0.0],
        [ 0.57968883, -0.8141048,   0.03455758, 0.0],
        [ 0.08653878,  0.10368021,  0.99083876, 0.0],
        [ 0.0,         0.0,         0.0,        1.0]
    ])


def create_toolOMY_transform():
    """OMYend_T_crpend"""
    return np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # return np.array([
    #     [0.0, -1.0, 0.0, 0.0],
    #     [-1.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, -1.0, 0.0],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])



def create_base_transform():
    """so101_DH_to_CRPArm"""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def create_base_transform_crpomy():
    """CRPBase_to_OMYBase"""
    return np.array([
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


# def create_omy_base_transform():
#     """CRPArm_T_OMY"""
#     return np.array([
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0]
#     ])

def dh_transform(a, alpha, d, theta):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    return np.array([
        [cos_t, -sin_t*cos_a,  sin_t*sin_a, a*cos_t],
        [sin_t,  cos_t*cos_a, -cos_t*sin_a, a*sin_t],
        [0,      sin_a,        cos_a,       d],
        [0,      0,            0,           1]
    ])


def map_range(value: float, input_min: float, input_max: float, output_min: float, output_max: float) -> float:
    """将数值从输入范围映射到输出范围"""
    # return (value - input_min) / (input_max - input_min) * (output_max - output_min) + output_min
    if value >= 0:
        return (value / input_max) * output_max
    else:
        return (value / input_min) * output_min


def so101_to_radian(action: dict[str, float]) -> list:
    action = copy.deepcopy(action)
    for key in action:
        action[key] = -action[key]

    # 每个关节的范围映射配置
    ranges = {
        'shoulder_pan.pos': (-100, 100, -2.0943, 2.0943),
        'shoulder_lift.pos': (-100, 100, -1.91983, 1.91983),
        'elbow_flex.pos': (-100, 100, -1.57077, 1.7453),
        'wrist_flex.pos': (-100, 100, -1.91983, 1.91983),
        'wrist_roll.pos': (-100, 100, -2.4434, 2.4434),
        'gripper.pos': (1e-10, 100, 0, 1)
    }

    # 创建一个新的字典，将每个关节的编码器数值映射为角度
    action_in_radian = {}
    
    # 对每个关节的位置进行范围映射
    for key, value in action.items():
        if key in ranges:
            input_min, input_max, output_min, output_max = ranges[key]
            action_in_radian[key] = map_range(value, input_min, input_max, output_min, output_max)
    
    # 返回一个包含所有映射后角度
    return [action_in_radian[key] for key in action_in_radian]


def forward_kinematics_so101(joint_angles: list[float]) -> list[float]:
    """
    输入: joint_angles - 长度为5的关节角度列表
    输出: [x, y, z, roll, pitch, yaw] - 末端位姿
    """
    if len(joint_angles) != 5:
        raise ValueError("SO101需要5个关节角度")
    
    dh_params = create_so101_dh_params()
    for i in range(5):
        dh_params[i][3] = float(dh_params[i][3] + joint_angles[i])

    # 计算各关节变换矩阵
    T1 = dh_transform(*dh_params[0])
    T2 = dh_transform(*dh_params[1])
    T3 = dh_transform(*dh_params[2])
    T4 = dh_transform(*dh_params[3])
    T5 = dh_transform(*dh_params[4])


    # 总变换矩阵（基座到末端）
    T_total = T1 @ T2 @ T3 @ T4 @ T5

    # print(T_total)
    # print(T1 @ T2)

    # 提取位置和姿态
    x, y, z = T_total[0, 3], T_total[1, 3], T_total[2, 3]
    rotation_matrix = T_total[:3, :3]
    rot_obj = R.from_matrix(rotation_matrix)
    
    roll, pitch, yaw = rot_obj.as_euler('xyz', degrees=False)
    
    # print("末端位置 (X, Y, Z):", x, y, z)

    return np.round([x, y, z, roll, pitch, yaw], 10).tolist()


def forward_kinematics(joint_angles: list[float]) -> list[float]:
    """
    获取so101末端位姿,并准换到CrpArmBase坐标系下
    输入: joint_angles - 长度为5的关节角度列表
    输出: [x, y, z, roll, pitch, yaw] - 末端位姿
    """
    if len(joint_angles) != 5:
        raise ValueError("需要5个关节角度")
    
    dh_params = create_so101_dh_params()
    for i in range(5):
        dh_params[i][3] = float(dh_params[i][3] + joint_angles[i])

    # 计算各关节变换矩阵
    T1 = dh_transform(*dh_params[0])
    T2 = dh_transform(*dh_params[1])
    T3 = dh_transform(*dh_params[2])
    T4 = dh_transform(*dh_params[3])
    T5 = dh_transform(*dh_params[4])
    
    Ttool = create_tool_transform()
    Tbase = create_base_transform()

    # 总变换矩阵（基座到末端）
    T_total = Tbase @ T1 @ T2 @ T3 @ T4 @ T5 @ Ttool

    # print("T_total",T_total)
    # print(T1 @ T2)

    # 提取位置和姿态
    x, y, z = T_total[0, 3], T_total[1, 3], T_total[2, 3]
    rotation_matrix = T_total[:3, :3]
    rot_obj = R.from_matrix(rotation_matrix)
    
    roll, pitch, yaw = rot_obj.as_euler('xyz', degrees=False)
    
    # print("末端位置 (X, Y, Z):", x, y, z)
    # print("末端姿态 (roll, pitch, yaw):", roll, pitch, yaw)

    return np.round([x, y, z, roll, pitch, yaw], 10).tolist()


def linear_map(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """标准线性映射"""
    if in_max == in_min:
        return out_min
    return (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def map_so2crp(end_pose: list[float]) -> list[float]:
    """
    输入: so101 - [x, y, z, roll, pitch, yaw]
    输出: CRPArm - [x, y, z, roll, pitch, yaw]
    
    位置映射: 从 SO101 的位置范围映射到 CRPArm 的位置范围
    姿态转换: SO101 使用弧度，CRPArm 使用角度
    """
    if len(end_pose) != 6:
        raise ValueError("end_pose 必须包含6个值: [x, y, z, roll, pitch, yaw]")
    
    x, y, z, roll, pitch, yaw = end_pose
    
    # SO101 位置范围 m
    so101_x_range = (0.05, 0.25)  # (min, max)
    so101_y_range = (-0.25, 0.25)  # (min, max)
    so101_z_range = (-0.10, 0.20)   # (min, max)
    
    # CRPArm 位置范围 mm
    crp_x_range = (320, 760)    # (min, max)
    crp_y_range = (-190, 400)    # (min, max)
    crp_z_range = (-315, 270)     # (min, max)
    
    # 限制输入值到 SO101 的范围内，确保安全
    x_clipped = np.clip(x, so101_x_range[0], so101_x_range[1])
    y_clipped = np.clip(y, so101_y_range[0], so101_y_range[1])
    z_clipped = np.clip(z, so101_z_range[0], so101_z_range[1])
    
    x_mapped = linear_map(x_clipped, so101_x_range[0], so101_x_range[1], crp_x_range[0], crp_x_range[1])
    y_mapped = linear_map(y_clipped, so101_y_range[0], so101_y_range[1], crp_y_range[0], crp_y_range[1])
    z_mapped = linear_map(z_clipped, so101_z_range[0], so101_z_range[1], crp_z_range[0], crp_z_range[1])
    
    # 限制输出值到 CRPArm 的范围内，确保安全
    x_mapped = np.clip(x_mapped, crp_x_range[0], crp_x_range[1])
    y_mapped = np.clip(y_mapped, crp_y_range[0], crp_y_range[1])
    z_mapped = np.clip(z_mapped, crp_z_range[0], crp_z_range[1])
    
    # 转换姿态: 从弧度转换为角度
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    # return np.round([x_mapped, y_mapped, z_mapped, 179.969, -0.024, -123.208], 10).tolist()
    # return np.round([490, 104, 217, roll_deg, pitch_deg, yaw_deg], 10).tolist()

    return np.round([x_mapped, y_mapped, z_mapped, roll_deg, pitch_deg, yaw_deg], 10).tolist()









def get_so101_endpose(action: dict[str, float]):
    action_copy = copy.deepcopy(action)
    joints_radian = so101_to_radian(action_copy)
    return forward_kinematics_so101(joints_radian[:5])


def get_endpose2Crp(action: dict[str, float]):
    action_copy = copy.deepcopy(action)
    joints_radian = so101_to_radian(action_copy)
    return map_so2crp(forward_kinematics(joints_radian[:5]))













def map_omy2crp(end_pose: list[float]) -> list[float]:
    """
    输入: OMY_L100 - [x, y, z, roll, pitch, yaw]毫米角度
    输出: CRPArm - [x, y, z, roll, pitch, yaw]毫米角度
    
    映射: 从 OMY_L100 的范围映射到 CRPArm 的范围
    """
    if len(end_pose) != 6:
        raise ValueError("end_pose 必须包含6个值: [x, y, z, roll, pitch, yaw]")
    
    x, y, z, roll_deg, pitch_deg, yaw_deg = end_pose

    # OMY_L100 位置范围 m
    omy_x_range = (40, 470)  # (min, max)
    omy_y_range = (-330, 330)  # (min, max)
    omy_z_range = (250, 700)   # (min, max)
    
    # CRPArm 位置范围 mm
    crp_x_range = (320, 800)    # (min, max)
    crp_y_range = (-400, 400)    # (min, max)
    crp_z_range = (-300, 300)     # (min, max)
    
    # 限制输入值到 OMY_L100 的范围内，确保安全
    x_clipped = np.clip(x, omy_x_range[0], omy_x_range[1])
    y_clipped = np.clip(y, omy_y_range[0], omy_y_range[1])
    z_clipped = np.clip(z, omy_z_range[0], omy_z_range[1])

    x_mapped = linear_map(x_clipped, omy_x_range[0], omy_x_range[1], crp_x_range[0], crp_x_range[1])
    y_mapped = linear_map(y_clipped, omy_y_range[0], omy_y_range[1], crp_y_range[0], crp_y_range[1])
    z_mapped = linear_map(z_clipped, omy_z_range[0], omy_z_range[1], crp_z_range[0], crp_z_range[1])
    
    # 限制输出值到 CRPArm 的范围内，确保安全
    x_mapped = np.clip(x_mapped, crp_x_range[0], crp_x_range[1])
    y_mapped = np.clip(y_mapped, crp_y_range[0], crp_y_range[1])
    z_mapped = np.clip(z_mapped, crp_z_range[0], crp_z_range[1])

    return np.round([x_mapped, y_mapped, z_mapped, roll_deg, pitch_deg, yaw_deg], 10).tolist()

 
def get_omy_endpose2Crp(end_pose):

    required_keys = {"ee.x", "ee.y", "ee.z", "ee.roll", "ee.pitch", "ee.yaw"}
    if not required_keys.issubset(set(end_pose.keys())):
        raise ValueError(f"字典必须包含键: {sorted(required_keys)}")

    try:
        x = float(end_pose["ee.x"])
        y = float(end_pose["ee.y"])
        z = float(end_pose["ee.z"])
        roll_deg = float(end_pose["ee.roll"])
        pitch_deg = float(end_pose["ee.pitch"])
        yaw_deg = float(end_pose["ee.yaw"])

    except Exception as e:
        raise ValueError(f"无法从字典中解析数值: {e}")

    # 将输入位姿（基坐标下的 OMY 末端）转为齐次坐标矩阵
    T_omy = np.eye(4)
    T_omy[:3, :3] = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    T_omy[:3, 3] = [x, y, z]


    T_extra1 = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    T_extra2 = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


    T_crpomy = create_base_transform_crpomy()
    T_tool_omy = create_toolOMY_transform()
    # T_result = T_crpomy @ T_omy @ T_tool_omy
    # T_result = T_extra1 @ T_crpomy @ T_omy @ T_tool_omy
    T_result = T_extra1 @ T_crpomy @ T_omy @ T_extra2 @ T_tool_omy



    # 从结果矩阵提取六个值 [x, y, z, roll, pitch, yaw]（角度为度）
    # x_new, y_new, z_new = T_result[0, 3], T_result[1, 3], T_result[2, 3]
    rot_obj = R.from_matrix(T_result[:3, :3])
    roll_new, pitch_new, yaw_new = rot_obj.as_euler("xyz", degrees=True)

    # 位置姿态矩阵不同
    return map_omy2crp([x-60, y+105, z, roll_new, pitch_new, yaw_new])
    # return map_omy2crp([x, y, z, -180, 0, 90])








def euler_to_rotation_matrix(euler_angles: list[float], seq: str = 'xyz', degrees: bool = False) -> np.ndarray:
    if not isinstance(euler_angles, (list, tuple)):
        raise ValueError("euler_angles must be a list or tuple of three numbers")
    if len(euler_angles) != 3:
        raise ValueError("euler_angles must have length 3: [a, b, c]")

    rot = R.from_euler(seq, euler_angles, degrees=degrees)
    Rmat = rot.as_matrix()
    T = np.eye(4, dtype=float)
    T[:3, :3] = Rmat
    return T

def get_world_T_so101end(action: dict[str, float])-> np.ndarray:
    action_copy = copy.deepcopy(action)
    joints_radian = so101_to_radian(action_copy)
    joints_radian_5 = joints_radian[:5]
    if len(joints_radian_5) != 5:
        raise ValueError("需要5个关节角度")
    
    dh_params = create_so101_dh_params()
    for i in range(5):
        dh_params[i][3] = float(dh_params[i][3] + joints_radian_5[i])

    T1 = dh_transform(*dh_params[0])
    T2 = dh_transform(*dh_params[1])
    T3 = dh_transform(*dh_params[2])
    T4 = dh_transform(*dh_params[3])
    T5 = dh_transform(*dh_params[4])

    Tbase = create_base_transform()

    T_total = Tbase @ T1 @ T2 @ T3 @ T4 @ T5
    return T_total




if __name__ == "__main__":

    # test_joints = [0.0, 0.0, 0.0, 0.0, np.pi/2]
    # end_pose = forward_kinematics(test_joints)
    # print("末端位置 (X, Y, Z):", end_pose[:3])
    # print("末端姿态 (roll, pitch, yaw):", end_pose[3:])



    # dict = {'shoulder_pan.pos': -9.344082081348475, 
    #         'shoulder_lift.pos': -29.158025715470757, 
    #         'elbow_flex.pos': 72.99729972997301, 
    #         'wrist_flex.pos': 5.586353944562902, 
    #         'wrist_roll.pos': 11.040645719227456, 
    #         'gripper.pos': 1.7651573292402147}
    
    # # _ = get_so101_endpose(dict)

    # A = get_endpose2Crp(dict)
    # print(A)

    print(create_tool_transform())
