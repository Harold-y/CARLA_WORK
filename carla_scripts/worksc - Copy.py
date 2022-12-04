import argparse
import glob
import os
import random
import sys
import time
import numpy as np
import cv2
import random
import pygame
import torch
import torchvision.transforms as transforms
import copy
import detr.util.misc as utils

from detr.models import build_model
from detr.datasets.kitti import make_kitti_transforms

import matplotlib.pyplot as plt
import time

try:
    import queue
except ImportError:
    import Queue as queue

set_client_map = 0
set_construction_cones = 1
set_construction_vehicle = 1
set_camera = 1
take_photo = 0
spawn_vehicle = 1
spawn_vehicle_num = 70
visualize_construction_zone = 1
visualize_detection_zone = 1
detection_range = 10
visualize_communication = 1
set_weather = 0
tst = 0

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

actor_list = []
vehicle_list = []
cone_list = []

CLASSES = [
    'Misc',
    'Cyclist',
    'Van',
    'Car',
    'Person_sitting',
    'Pedestrian',
    'Tram',
    'Truck',
    'Dont Care',
    'Dont Care',
    'Dont Care'
]
conf_thres = 0.7
transform = transforms.Compose([
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)
detr.num_queries = 10
detr.class_embed.weight = torch.nn.Parameter(data=torch.rand((16, 256)))
detr.class_embed.bias = torch.nn.Parameter(data=torch.rand(16))
detr.query_embed.weight = torch.nn.Parameter(data=torch.rand((10, 256)))
checkpoint = torch.load('F:\Programming\detr\output\checkpoint.pth', map_location='cpu')
detr.load_state_dict(checkpoint['model'])
detr.eval()
detr.to(device)
'''
args = {'aux_loss': True, 'backbone': 'resnet50', 'batch_size': 6, 'bbox_loss_coef': 5, 'clip_max_norm': 0.1,
        'data_panoptic_path': None,
        'data_path': 'D:\\Programming\\PycharmProjects\\resources\\Kitti\\Kitti\\raw\\testing\\image_2\\',
        'dataset_file': 'kitti', 'dec_layers': 6, 'device': 'cuda', 'dice_loss_coef': 1, 'dilation': False,
        'dim_feedforward': 2048, 'dropout': 0.1, 'enc_layers': 6, 'eos_coef': 0.1, 'epochs': 300,
        'frozen_weights': None, 'giou_loss_coef': 2, 'hidden_dim': 256,
        'lr': 0.0001, 'lr_backbone': 1e-05, 'lr_drop': 200, 'mask_loss_coef': 1, 'masks': False, 'nheads': 8,
        'num_queries': 10, 'output_dir': '', 'position_embedding': 'sine', 'pre_norm': False, 'remove_difficult': False,
        'resume': "output\\checkpoint.pth",
        'set_cost_bbox': 5, 'set_cost_class': 1, 'set_cost_giou': 2, 'thresh': 0.5, 'weight_decay': 0.0001}
detr, _, postprocessors = build_model(argparse.Namespace(**args))
checkpoint = torch.load("detr/output/checkpoint.pth", map_location='cpu')
detr.load_state_dict(checkpoint['model'])
detr.to(device)


def destroy():
    for actor in actor_list:
        actor.destroy()
    for cone in cone_list:
        cone.destroy()


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_bbox(img_bgr, pred):
    # create a copy of image for bounding-box visualization
    img_copy = img_bgr.copy()

    # Extract class probability and bounding-box
    pred_logits = pred['pred_logits'][0]
    pred_boxes = pred['pred_boxes'][0]

    # get softmax of logits and choose the index with highest probability
    pred_prob = pred_logits.softmax(-1)
    pred_prob_np = pred_prob.cpu().numpy()
    pred_idx = np.argmax(pred_prob_np, -1)

    ## Filter out detections and draw bounding-box
    # iterate through predictions
    for i, (idx, box) in enumerate(zip(pred_idx, pred_boxes)):
        if (idx >= len(CLASSES)) or (pred_prob_np[i][idx] < conf_thres):
            continue

        label = CLASSES[idx]
        box = box.cpu().numpy() * [img_bgr.shape[1], img_bgr.shape[0], img_bgr.shape[1], img_bgr.shape[0]]
        x, y, w, h = box

        # get bbox corners
        x0, y0, x1, y1 = int(x - (w / 2)), int(y - (h / 2)), int(x + (w / 2)), int(y + (h / 2))

        # draw bounding-box
        cv2.rectangle(img_copy, (x0, y0), (x1, y1), thickness=2, color=(0, 0, 255))

        # A rectangle behind text
        cv2.rectangle(img_copy, (x0, y0 - 20), (x0 + 100, y0), color=(0, 0, 0), thickness=-1)

        # write object id
        cv2.putText(img_copy, '{} {:.1f}%'.format(label, pred_prob_np[i][idx] * 100.0),
                    (x0 + 5, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=2)

    # return the image with bounding-box
    return img_copy


def draw_image(surface, image, blend=False):
    global tst

    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array2 = copy.deepcopy(array)

    img_tensor = transform(array2).unsqueeze(0).cuda()

    with torch.no_grad():
        pred = detr(img_tensor)
    array = draw_bbox(array2, pred)

    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    if set_client_map == 1:
        world = client.load_world('Town02')
    else:
        world = client.get_world()
    pygame.init()

    display = pygame.display.set_mode(
        (1200, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()
    spawn_points = world.get_map().get_spawn_points()
    blueprint_library = world.get_blueprint_library()

    spectator = world.get_spectator()
    spectator.set_transform(
        carla.Transform(carla.Location(x=-65, y=17, z=10), carla.Rotation(pitch=-20, yaw=30, roll=0)))

    if set_camera == 1:
        IM_WIDTH = 1200
        IM_HEIGHT = 600
        camera1 = blueprint_library.find('sensor.camera.rgb')
        # Change the dimensions of the image
        camera1.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera1.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera1.set_attribute('fov', '110')

        spawn_point_cam = carla.Transform(carla.Location(x=-65, y=17, z=10), carla.Rotation(pitch=-20, yaw=30, roll=0))

        actor_camera1 = world.spawn_actor(blueprint=camera1, transform=spawn_point_cam)
        actor_list.append(actor_camera1)

    if set_weather == 1:
        weather = world.get_weather()
        weather.sun_altitude_angle = 10
        weather.cloudiness = 10
        weather.precipitation = 50
        weather.precipitation_deposits = 60
        weather.fog_density = 60
        weather.fog_distance = 7
        world.set_weather(weather)

    if spawn_vehicle == 1:
        for i in range(spawn_vehicle_num):
            # Choose random blueprint and choose the i-th default spawn points
            vehicle_bp_i = random.choice(blueprint_library.filter('vehicle.*.*'))
            spawn_point_i = spawn_points[i]
            print(spawn_point_i.location, spawn_point_i.rotation)
            # Spawn the actor
            vehicle_i = world.try_spawn_actor(vehicle_bp_i, spawn_point_i)

            # Append to the actor_list
            if vehicle_i != None:
                actor_list.append(vehicle_i)
                vehicle_list.append(vehicle_i)
        print('%d vehicles are generated' % len(vehicle_list))

        # Set autopilot for each vehicle
        for vehicle_i in vehicle_list:
            vehicle_i.set_autopilot(True)

    with CarlaSyncMode(world, actor_camera1, fps=10) as sync_mode:
        while True:
            if should_quit():
                exit(0)
            clock.tick()

            # Advance the simulation and wait for the data.
            snapshot, image_rgb = sync_mode.tick(timeout=2.0)

            fps = round(1.0 / snapshot.timestamp.delta_seconds)

            # Draw the display.
            draw_image(display, image_rgb)
            display.blit(
                font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                (8, 10))
            display.blit(
                font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                (8, 28))
            pygame.display.flip()


finally:
    destroy()
