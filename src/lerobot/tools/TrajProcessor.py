import math
from typing import List

class TrajectoryProcessor:

    def __init__(self, max_points: int = 1, max_joints: int = 5):
        if max_points < 1:
            raise ValueError("max_points must be >= 1")
        self.max_points = int(max_points)
        self.max_joints = int(max_joints)

        self._written_once_point = False
        self._written_once_joint = False
        
        self.points: List[List[float]] = [[0.0] * 6 for _ in range(self.max_points)]
        self.joints: List[List[float]] = [[0.0] * 6 for _ in range(self.max_joints)]

    def trajectory_differential(self, current_pose: List[float], target_pose: List[float], step_length: float = 0.1) -> List[float]:
        """Compute one incremental pose step from current_pose towards target_pose.

        The translation (x,y,z) moves by at most step_length along the straight line.
        Orientation (roll,pitch,yaw) is taken from the target_pose.
        """
        # Input validation
        if not (isinstance(current_pose, (list, tuple)) and isinstance(target_pose, (list, tuple))):
            raise ValueError("current_pose and target_pose must be list or tuple of length 6")
        if len(current_pose) != 6 or len(target_pose) != 6:
            raise ValueError("current_pose and target_pose must have length 6: [x,y,z,roll,pitch,yaw]")
        if step_length < 0:
            raise ValueError("step_length must be non-negative")

        cx, cy, cz = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
        tx, ty, tz = float(target_pose[0]), float(target_pose[1]), float(target_pose[2])

        dx = tx - cx
        dy = ty - cy
        dz = tz - cz

        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        # If already at or within step_length, return the target pose
        if dist <= step_length or dist == 0.0:
            return [float(x) for x in target_pose]

        # Move along the unit vector by step_length
        ux = dx / dist
        uy = dy / dist
        uz = dz / dist

        new_x = cx + ux * step_length
        new_y = cy + uy * step_length
        new_z = cz + uz * step_length

        # Use target orientation
        new_roll = float(target_pose[3])
        new_pitch = float(target_pose[4])
        new_yaw = float(target_pose[5])

        return [round(new_x, 10), round(new_y, 10), round(new_z, 10), new_roll, new_pitch, new_yaw]
    

    
    def write_point(self, vec: List[float]) -> None:
        """Write a single 6-element vector into the points history.

        Keeps at most `max_points` latest entries. The buffer order is oldest->newest.
        """
        if not (isinstance(vec, (list, tuple)) and len(vec) == 6):
            raise ValueError("vec must be a list/tuple of length 6")
        point = [float(x) for x in vec]

        if not self._written_once_point:
            # First real write: fill all slots with this point
            self.points = [point[:] for _ in range(self.max_points)]
            self._written_once_point = True
            return

        # Subsequent writes: append and keep only last max_points
        self.points.append(point)
        if len(self.points) > self.max_points:
            self.points = self.points[-self.max_points:]

    def read_points(self) -> List[List[float]]:
        """Return a shallow copy of the stored points list."""
        return [p[:] for p in self.points]
    


    def write_joint(self, vec: List[float]) -> None:
        """Write a single 6-element vector into the joints history.

        Keeps at most `max_joints` latest entries. The buffer order is oldest->newest.
        """
        if not (isinstance(vec, (list, tuple)) and len(vec) == 6):
            raise ValueError("vec must be a list/tuple of length 6")
        joint = [float(x) for x in vec]

        if not self._written_once_joint:
            # First real write: fill all slots with this joint
            self.joints = [joint[:] for _ in range(self.max_joints)]
            self._written_once_joint = True
            return

        # Subsequent writes: append and keep only last max_joints
        self.joints.append(joint)
        if len(self.joints) > self.max_joints:
            self.joints = self.joints[-self.max_joints:]

    def read_joints(self) -> List[List[float]]:
        """Return a shallow copy of the stored joints list."""
        return [j[:] for j in self.joints]


    def init_matrix(self, flat: List[float], group_size: int) -> List[List[float]]:
        if not isinstance(flat, (list, tuple)):
            raise ValueError("flat must be a list or tuple of numbers representing a single vector")
        if not isinstance(group_size, int) or group_size < 1:
            raise ValueError("group_size must be an int >= 1")

        if len(flat) == 0:
            return []

        row = [float(x) for x in flat]
        return [row[:] for _ in range(group_size)]


if __name__ == "__main__":
    
    # 1. 初始化测试
    print("\n[1] Init test")
    tp = TrajectoryProcessor(max_points=3)
    print("Initial points:", tp.read_points())

    # 2. trajectory_differential 测试
    print("\n[2] trajectory_differential test")

    current_pose = [0, 0, 0, 0, 0, 0]
    target_pose = [1, 0, 0, 10, 20, 30]
    step = 0.2

    next_pose = tp.trajectory_differential(
        current_pose=current_pose,
        target_pose=target_pose,
        step_length=step
    )

    print("Current pose:", current_pose)
    print("Target pose :", target_pose)
    print("Next pose   :", next_pose)

    # 期望：x 增加 0.2，其它位置不变，姿态直接等于目标
    # [0.2, 0.0, 0.0, 10, 20, 30]

    # 3. trajectory_differential：小于 step_length 情况
    print("\n[3] trajectory_differential near-target test")

    current_pose = [0.95, 0, 0, 0, 0, 0]
    next_pose = tp.trajectory_differential(
        current_pose=current_pose,
        target_pose=target_pose,
        step_length=step
    )
    print("Near target result:", next_pose)

    # 期望：直接返回 target_pose

    # 4. write_point 测试
    print("\n[4] write_point test")

    tp.write_point([1, 2, 3, 4, 5, 6])
    print("After first write:", tp.read_points())

    tp.write_point([2, 3, 4, 5, 6, 7])
    print("After second write:", tp.read_points())

    tp.write_point([3, 4, 5, 6, 7, 8])
    print("After third write:", tp.read_points())

    tp.write_point([4, 5, 6, 7, 8, 9])
    print("After fourth write (FIFO):", tp.read_points())

    # 期望：
    # max_points = 3
    # 最终只保留最后 3 个写入点

    # 5. read_points 拷贝安全性测试
    print("\n[5] read_points copy safety test")

    points_copy = tp.read_points()
    points_copy[0][0] = 999  # 修改拷贝
    print("Modified copy :", points_copy)
    print("Internal points:", tp.read_points())

    # 期望：内部 points 不被修改