from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate,
    ScaraRobotTemplate,
    FiveDOFRobotTemplate,
)


class ScaraRobot(ScaraRobotTemplate):
    def __init__(self):
        super().__init__()

    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()

        th1, th2, d3 = curr_joint_values[0], curr_joint_values[1], curr_joint_values[2]
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5

        dh_table = np.array([[th1, l1, l2, 0], [th2, l3 - l5, l4, 0], [0, -d3, 0, 180]])
        Hlist = []

        for i in range(0, len(curr_joint_values)):
            thi = dh_table[i, 0]
            di = dh_table[i, 1]
            ai = dh_table[i, 2]
            alphai = dh_table[i, 3]
            H_matrix = np.array(
                [
                    [
                        cos(thi),
                        -sin(thi) * cos(alphai),
                        sin(thi) * sin(alphai),
                        ai * cos(thi),
                    ],
                    [
                        sin(thi),
                        cos(thi) * cos(alphai),
                        -cos(thi) * sin(alphai),
                        ai * sin(thi),
                    ],
                    [0, sin(alphai), cos(alphai), di],
                    [0, 0, 0, 1],
                ]
            )
            Hlist.append(H_matrix)
        [H0_1, H1_2, H2_3] = Hlist

        # Calculate EE position and rotation
        H_ee = H0_1 @ H1_2 @ H2_3  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]

        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist


if __name__ == "__main__":
    model = ScaraRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
