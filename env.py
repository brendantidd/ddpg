
import numpy as np
from numpy import cos as c
from numpy import sin as s
import pybullet as p
from gym import spaces
from itertools import chain
import time

MAX_MOTOR_POSITION = 0.5
TARGET_X = 0

def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item

class Env():
    def __init__(self):
        self.RENDER = True

        if self.RENDER:
            physicsClient = p.connect(p.GUI)
        else:
            physicsClient = p.connect(p.DIRECT)

        PLANE_PATH = '/home/brendan/bullet3/data/'
        # DATA_PATH = '/home/brendan/baselines/baselines/ddpg//'

        p.setGravity(0,0,-10)
        self.planeId = p.loadURDF(PLANE_PATH + "plane.urdf")
        # self.DT = 0.004
        self.DT = 0.001
        # self.DT = 0.01
        # self.DT = 0.1
        p.setTimeStep(self.DT)


        self.initial_pose = [0,0,1]
        self.pose = self.initial_pose
        self.prev_pose = [self.initial_pose]
        self.orien = [0,0,0]
        self.prev_orien = [0,0,0]

        self.initial_orientation = p.getQuaternionFromEuler([0,0,0])
        # self.Id = p.loadURDF(DATA_PATH + "assets/mybot.urdf", self.initial_pose, self.initial_orientation, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        self.Id = p.loadURDF("assets/mybot.urdf", self.initial_pose, self.initial_orientation)

        self.num_joints = p.getNumJoints(self.Id)
        # print(self.num_joints)
        self.all_joints = [p.getJointInfo(self.Id, i) for i in range(self.num_joints)]
        # print(self.all_joints)
        self.motor_names = ["right_hip_x_joint", "right_hip_z_joint", "right_hip_y_joint", "right_knee_joint", "right_ankle_x_joint", "right_ankle_y_joint"]
        self.motor_names += ["left_hip_x_joint", "left_hip_z_joint", "left_hip_y_joint", "left_knee_joint", "left_ankle_x_joint", "left_ankle_y_joint"]
        self.joints ={name:j[0] for j in self.all_joints for name in self.motor_names if j[1].decode() == name}
        self.joint_numbers = [self.joints[key] for key in self.joints]
        # print(self.joint_numbers)
        self.left_foot_sensors = [i[0] for i in self.all_joints if "left_foot_s" in i[1].decode()]
        self.right_foot_sensors = [i[0] for i in self.all_joints if "right_foot_s" in i[1].decode()]
        # print(self.left_foot_sensors, self.right_foot_sensors)
        high = MAX_MOTOR_POSITION*np.ones(12)
        low = -high
        self.action_space = spaces.Box(low, high)
        self.action_size = self.action_space.shape[0]

        high = np.inf*np.ones(36)
        low = -high
        self.observation_space = spaces.Box(low,high)
        self.state_size = self.observation_space.shape[0]

        self.reward_range = 10
        self.spec = None
        self.state_dict = {}
        self.inverse_state_dict = {}
        self.prev_state_dict = {}
        self.prev_inverse_state_dict = {}
        self.metadata = {
                'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': int(np.round(1.0 / self.DT))
        }

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, actions):
        assert( np.isfinite(actions).all() )
        # self.set_motors_velocity(actions)
        self.set_motors_position(actions)
        p.stepSimulation()
        self.get_state()

        if self.RENDER:
            time.sleep(self.DT)

        # calculate the reward
        self.reward = 2

        # Target position
        self.reward += (self.state_dict['x'] - TARGET_X)

        # Penalties:
        self.reward -= abs(self.state_dict['y'])
        self.reward -= (abs(self.state_dict['roll']) + abs(self.state_dict['yaw']) + abs(self.state_dict['pitch']))

        if self.state_dict['z'] < 0.65:
            self.done = True
            self.reward = -1
        else:
            self.done = False

        return list(flatten([self.state_dict[key] for key in self.state_dict])), self.reward, self.done, dict(reward_linvel=0, reward_quadctrl=0, reward_alive=0, reward_impact=0)

    def reset(self):
        self.previous_x = 0
        p.resetBasePositionAndOrientation(self.Id, self.initial_pose, self.initial_orientation)
        self.reward = 0
        self.done = False
        self.set_motors_position(np.zeros(self.action_size))
        p.stepSimulation()
        self.get_state()
        return list(flatten([self.state_dict[key] for key in self.state_dict]))

    def reset_joint_positions(self):
        for i in self.joint_numbers:
            p.resetJointState(self.Id, i, 0.0)

    def set_motors_velocity(self, actions):
        order = [self.joints[key] for key in self.motor_names]
        p.setJointMotorControlArray(self.Id,order,p.VELOCITY_CONTROL, actions)

    def set_motors_position(self, actions):
        order = [self.joints[key] for key in self.motor_names]
        p.setJointMotorControlArray(self.Id,order,p.POSITION_CONTROL, actions)

    def get_state(self):
        self.pose = p.getBasePositionAndOrientation(self.Id)
        self.orien = p.getEulerFromQuaternion(self.pose[1])
        self.state_dict = {'x': self.pose[0][0], 'y':self.pose[0][1], 'z':self.pose[0][2], 'roll': self.orien[0], 'pitch': self.orien[1], 'yaw': self.orien[2]}
        self.state_dict.update({'x_dot': (self.pose[0][0] - self.prev_pose[0][0])/self.DT ,'y_dot':(self.pose[0][1] - self.prev_pose[0][1])/self.DT, 'z_dot': (self.pose[0][2] - self.prev_pose[0][2])/self.DT})
        self.state_dict.update({'roll_dot': (self.orien[0]-self.prev_orien[0])/self.DT,'pitch_dot': (self.orien[1]-self.prev_orien[1])/self.DT, 'yaw_dot': (self.orien[2]-self.prev_orien[2])/self.DT})
        self.state_dict.update({i:(p.getJointState(self.Id, self.joints[i])[0], p.getJointState(self.Id, self.joints[i])[1]) for i in self.joints})
        self.prev_pose = self.pose
        self.prev_orien = self.orien
