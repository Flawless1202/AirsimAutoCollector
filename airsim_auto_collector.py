import os
import time
import copy
import random
from itertools import permutations

import numpy as np
import airsim
import simplejson as json
from easydict import EasyDict


class AutoCollector(object):

    def __init__(self, config_file):
        self.client = airsim.VehicleClient()
        self.client.confirmConnection()
        self.config = self.load_config(config_file)
        self.init_dirs()
        self.init_vehicle_pose()
        self.init_instance_ids()
        self.init_object_combinations()

    def load_config(self, config_file):
        with open(config_file) as f:
            config = EasyDict(json.load(f))
            return config

    def init_dirs(self):
        root_dir = self.config.root_dir
        os.makedirs(root_dir, exist_ok=True)
        for subdir in self.config.saved_img_types:
            os.makedirs(os.path.join(root_dir, subdir), exist_ok=True)

    def init_vehicle_pose(self):
        self.vehicle_init_pose = self.client.simGetObjectPose("PlayerStart")
        if np.isnan(self.vehicle_init_pose.position.x_val):
            print("Please set the tag of PlayerStart to 'PlayerStart in UE4!'")

    def init_instance_ids(self):
        instance_ids = self.config.instance_ids
        self.client.simSetSegmentationObjectID("[\w]*", 255, True)

        for key in instance_ids.keys():
            self.client.simSetSegmentationObjectID(key, instance_ids[key], True)

        for idx in range(1, self.config.object_types + 1):
            key = "obj{}[\w]*".format(idx)
            self.client.simSetSegmentationObjectID(key, idx, True)

    def init_object_combinations(self):
        single_object = list()
        
        for idx in range(1, self.config.object_types + 1):
            single_object.append(["obj{}".format(idx)])
        
        self.config.object_combinations = single_object + self.config.object_combinations

    def get_object_positions(self, object_combination):
        object_position_candidates = self.config.object_position_candidates
        object_positions_permutations = list(permutations(object_position_candidates,
                                                          len(object_combination)))
        position_idx = int(np.floor(np.random.rand() *
                                    len(object_positions_permutations)))
        object_positions = object_positions_permutations[position_idx]
        return dict(zip(object_combination, object_positions))

    def record_img(self):
        if len(self.config.record_poses) == 0:
            vehicle_end = self.config.vehicle_end
            record_num = self.config.record_num
            if vehicle_end[0] == 0:  # along x axis
                record_poses = [[pos, 0]
                                for pos in np.linspace(0, vehicle_end[1], record_num, endpoint=True)]
            else:  # along y axis
                record_poses = [[0, pos]
                                for pos in np.linspace(0, vehicle_end[1], record_num, endpoint=True)]
        else:
            record_poses = self.config.record_pos

        for object_combination in self.config.object_combinations:
            self.record_img_single(record_poses, object_combination)

    def record_img_single(self, record_poses, object_combination):
        object_positions = self.get_object_positions(object_combination)
        object_combination_str = "_".join(object_combination)

        ori_poses = dict()
        time_count = 0

        for record_pose in record_poses:

            for obj in object_positions:
                pose = self.client.simGetObjectPose(obj)

                while not self.is_position_right(pose.position):
                    pose = self.client.simGetObjectPose(obj)

                ori_poses[obj] = copy.deepcopy(pose)
                pose.position.x_val, pose.position.y_val = object_positions[obj]
                pose_roll, pose_pitch, pose_yaw = airsim.to_eularian_angles(
                    pose.orientation)
                pose_yaw += random.uniform(0, 1) * np.pi / 180 * self.config.object_orientation_err
                pose.orientation = airsim.to_quaternion(pose_roll, pose_pitch, pose_yaw)

                while not self.is_moved_successful(self.client.simGetObjectPose(obj), pose):
                    success = self.client.simSetObjectPose(obj, pose)
                    if not success:
                        print("Error occured when try to move combination {} object {}!".format(
                            object_combination_str, obj))

            new_pose = copy.deepcopy(self.vehicle_init_pose)
            new_pose.position.x_val, new_pose.position.y_val = (
                record_pose[0] + self.config.vehicle_position_err * random.uniform(-1, 1),
                record_pose[1] + self.config.vehicle_position_err * random.uniform(-1, 1)
            )

            new_pose_roll, new_pose_pitch, new_pose_yaw = airsim.to_eularian_angles(new_pose.orientation)
            new_pose_yaw += random.uniform(-1, 1) * np.pi / 180 * self.config.vehicle_orientation_err
            new_pose.orientation = airsim.to_quaternion(new_pose_roll, new_pose_pitch, new_pose_yaw)

            self.client.simSetVehiclePose(new_pose, ignore_collison=True)
            responses = self.client.simGetImages([
                airsim.ImageRequest(0, airsim.ImageType.Scene, False, True),
                airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, True),
                airsim.ImageRequest(0, airsim.ImageType.DepthVis, False, True)
            ])

            record_time_stamp = responses[0].time_stamp

            for response in responses:
                if time_count == 0:
                    break
                airsim.write_file(os.path.join(self.config.root_dir, self.config.img_types[response.image_type],
                                               "{}.png".format(record_time_stamp)), response.image_data_uint8)

            time_count += 1

            for obj in object_positions:
                while not self.is_moved_successful(self.client.simGetObjectPose(obj), ori_poses[obj]):
                    success = self.client.simSetObjectPose(obj, ori_poses[obj])
                    if not success:
                        print("Error occured when try to move combination {} object {}!".format(
                            object_combination_str, obj))

    def is_moved_successful(self, current_pose, target_pose):
        current_position = current_pose.position
        target_position = target_pose.position
        return abs(current_position.x_val - target_position.x_val) < 0.1 and abs(current_position.y_val - target_position.y_val) < 0.1

    def is_position_right(self, position):
        return not (np.isnan(position.x_val) or np.isnan(position.x_val) or np.isnan(position.x_val))


if __name__ == "__main__":
    collector = AutoCollector("./collector_config.json")
    collector.record_img()
