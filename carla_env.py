import carla
import random
import time

import gym
import numpy as np
import cv2
import math

from gym import spaces
from stable_baselines3.common.env_checker import check_env

SECONDS_PER_EPISODE = 10
SHOW_PREVIEW = False
IM_WIDTH = 100   #1204
IM_HEIGHT = 100    #320

class CarEnv(gym.Env):
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self, seconds_per_episode=None, playing=False):
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(200.0)
        # self.world = self.client.load_world('Town06')
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self._max_episode_steps = 50

        self.action_space = spaces.Box(low=np.array([0.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, self.im_height, self.im_width), dtype=np.uint8)
    def reset(self):
        # def reset(self, i):
        self.world=self.client.reload_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

        self.collision_hist = []
        self.actor_list = []

        # 手动设置生成点
        self.location = carla.Location(20, 5, 5)
        self.rotation = carla.Rotation(0, 0, 0)
        self.transform = carla.Transform(self.location, self.rotation)

        # 自动选择生成点
        #self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")
        # 设置传感器捕捉数据时间间隔秒
        self.rgb_cam.set_attribute('sensor_tick', '1.0')
        # 设置相机相对依附位置
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        # 保存图片
        # cc = carla.ColorConverter.Raw
        # self.sensor.listen(lambda data: data.save_to_disk('_out/%06d.png' % data.frame_number, cc))
        # finsh
        # self.sensor.listen(lambda data: self.process_img(data))

        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # 这里的解决方法是，一旦我们准备好开始一个episode，就同时踩下油门和刹车，然后松开刹车。
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        time.sleep(20)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # 松开制动器
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))
        print(self.front_camera.shape)
        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        # i2 = i.reshape((4, self.im_height, self.im_width))
        # i3 = i2[:3, :, :]
        i3 = i2[:, :, :3]
        i3 = i3.reshape((3, self.im_height, self.im_width))
        # 深度图
        i4 = i2[:, :, :]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            # cv2.imwrite("")
            cv2.waitKey(1)
        self.front_camera = i3
        # return self.front_camera

    # def step(self, action):
    #     if action == 0:
    #         self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))
    #     elif action == 1:
    #         self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
    #     elif action == 2:
    #         self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))


    # 0: 'left', 1: 'forward', 2: 'right', 3: 'forward_left', 4: 'forward_right',
    # 5: 'brake', 6: 'brake_left', 7: 'brake_right'
    def step(self, action):
        info = {}
        print(action)
        if action[0] < 0:
            action[0] = 0.0
        action[0] = 1.0
        if action[2] < 0:
            action[2] = 0.0
        action[2] = 0.0
        # if action == 0:  # 直行
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0, brake=0))
        # elif action == 1: # 左转
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-0.5 * self.STEER_AMT, brake=0))
        # elif action == 2: # 右转
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0.5 * self.STEER_AMT, brake=0))
        # elif action == 3:  # 加速左转
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-0.5 * self.STEER_AMT, brake=0))
        # elif action == 4:  # 加速右转
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.5 * self.STEER_AMT, brake=0))
        # elif action == 5:  # 减速左转
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-0.5 * self.STEER_AMT, brake=0.5))
        # elif action == 6:  # 减速右转
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0.5 * self.STEER_AMT, brake=0.5))
        # elif action == 7:  # 刹车
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1.0))
        self.vehicle.apply_control(carla.VehicleControl(throttle=float(action[0]), steer=float(action[1]), brake = float(action[2])))



        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        location = self.vehicle.get_location()
        dis = int(math.sqrt(location.x ** 2 + location.y ** 2 + location.z ** 2))
        if len(self.collision_hist) != 0:
            done = True
            reward = -1
        elif kmh < 1:
            done = False
            reward = -1
        else:
            done = False
            reward = dis
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        #print(reward)
        return self.front_camera, reward, done, info

FPS = 30

# env = CarEnv()
# check_env(env)

# sate = env.reset()
# while True:
#
#     action = np.random.randint(0, 3)
#     new_state, reward, done, _ = env.step(action)
#
#     time.sleep(1 / FPS)
#
#     if done:
#         break