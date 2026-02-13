import random

import open3d as o3d
import numpy as np
import carla
import os
import math
from matplotlib import cm
from math import sin, cos, radians
from open3d.examples.visualization.draw import time_animation


def set_sync_mode(world,client,simulation_time_for_tick):
    """
    Set the world to synchronous mode.
    """
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = simulation_time_for_tick
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

def set_async_mode(world,client):
    """
    Set the world to asynchronous mode.
    """
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(False)

def destroy_all_vehicles(world:carla.World,client:carla.Client):
    """
    Destroy all vehicles in the world.
    """
    print("Destroying all vehicles...")
    vehicle_alive = world.get_actors().filter('vehicle.*')
    client.apply_batch([carla.command.DestroyActor(vehicle) for vehicle in vehicle_alive])
    print("Destroying all vehicles... DONE.")

def destroy_all_pedestrians(world:carla.World,client:carla.Client):
    """
    Destroy all the pedestrians
    """
    print("Destroying all pedestrians...")
    pedestrian_alive = world.get_actors().filter('walker.*')
    contr = world.get_actors().filter("controller*")
    for controller in contr:
        controller.stop()
    client.apply_batch([carla.command.DestroyActor(controller.id) for controller in contr])
    client.apply_batch([carla.command.DestroyActor(pedestrian.id) for pedestrian in pedestrian_alive])
    print("Destroying all pedestrians... DONE.")

def destroy_all_sensors(world:carla.World,client:carla.Client, sensors):
    """
    Destroy all sensors in the world.
    """
    print("Destroying all sensors...")
    # sensors is a list of class Sensor, the destroy method calls the stop and destroy for each sensor
    for x in sensors:
        x.destroy()
    print("Destroying all sensors... DONE.")

def destroy_all_anomalies(world:carla.World,client:carla.Client):
    """
    Destroy all anomalies in the world.
    """
    print("Destroying all anomalies...")
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter("blueprint.*")])
    print("Destroying all anomalies... DONE.")

def clean_up(args,world,client,tm,sensors):
    """
    Clean up the world by destroying all actors.
    """
    print("Cleaning Simulation...")
    destroy_all_sensors(world, client, sensors)
    destroy_all_anomalies(world, client)
    destroy_all_vehicles(world, client)
    destroy_all_pedestrians(world, client)
    set_async_mode(world, client)
    if args.hybrid:
        print("Deactivating hybrid mode")
        tm.set_hybrid_physics_mode(False)
    print("Cleaning Simulation... DONE.")

def load_map(world,client,map_name):
    """Load a map in CARLA simulator.
    Args:
        world (carla.World): The CARLA world object.
        map_name (str): The name of the map to load.
    """
    # Load the map
    world = client.load_world(map_name)
    print(f"Loading map: {map_name}")
    print(f"Map {map_name} loaded successfully.")

    # Set the spectator to a default location
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(carla.Location(x=0, y=0, z=50), carla.Rotation(pitch=-90)))
    return world

def spawn_ego_vehicle(world,client,args):
    """Spawn an ego vehicle in the CARLA simulator.
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
    """
    # Get the blueprint library
    blueprint_library = world.get_blueprint_library()

    # Get the vehicle blueprints and spawn points
    vehicle_blueprints = blueprint_library.filter("*mustang*")
    if args.spawn_points is not None:
        spawn_points = args.spawn_points
    else:
        spawn_points = world.get_map().get_spawn_points()
    # Choose a random vehicle blueprint and spawn point
    vehicle_blueprint:carla.ActorBlueprint = random.choice(vehicle_blueprints)
    vehicle_blueprint.set_attribute("role_name", "hero")
    ego_color = random.choice(vehicle_blueprint.get_attribute('color').recommended_values)
    vehicle_blueprint.set_attribute('color', ego_color)
    spawn_point = random.choice(spawn_points)
    # Spawn the vehicle
    vehicle = world.try_spawn_actor(vehicle_blueprint, spawn_point)
    if vehicle is None:
        print("Failed to spawn vehicle. Probably the spawn point is not valid.")
    else:
        # Needed to advance the simulation to set physics
        world.tick()
        print("Vehicle spawned successfully at location:", vehicle.get_transform())
    return vehicle

def generate_traffic(args,world,client):
    """Generate traffic in the CARLA simulator.
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
    """
    bp_vehicles = world.get_blueprint_library().filter(args.filterv)
    spawn_points = world.get_map().get_spawn_points()
    number_of_vehicles = args.number_of_vehicles
    if len(spawn_points) < number_of_vehicles:
        number_of_vehicles = len(spawn_points)
    random.shuffle(spawn_points)
    vehicle_list = []
    batch = []
    for i in range(number_of_vehicles):
        vehicle_bp = random.choice(bp_vehicles)
        while vehicle_bp.id == "vehicle.nissan.patrol" or vehicle_bp.id == "vehicle.ambulance.ford":
            # Nissan Patrol we skip it
            vehicle_bp = random.choice(bp_vehicles)
        vehicle_bp.set_attribute('role_name', 'autopilot')
        if vehicle_bp.has_attribute('color'):
            ego_color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color', ego_color)
        # SpawnActor is used to spawn the actor, .then is used to give it a command after spwawning the actor. FutureActor is used to get the actor that was just spawned.
        batch.append(carla.command.SpawnActor(vehicle_bp, spawn_points[i]))
    # Apply_batch_sync is used to apply the batch in just one simulation tick. This is faster and less comp. intensive than spawning one vehicle at a time.
    res = client.apply_batch_sync(batch,True)
    for response in res:
        if response.error:
            print("Generate Traffic: ", response.error)
        else:
            vehicle_list.append(response.actor_id)
    return vehicle_list

def generate_pedestrian(args,world,client):
    """Generate pedestrians in the CARLA simulator.
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
    """
    bp_pedestrians = world.get_blueprint_library().filter(args.filterw)
    # 1. Get the spawn points
    spawn_points=[]
    for i in range(args.number_of_pedestrians):
        spawn_point = carla.Transform()
        #get_random_location_from_navigation() is used to get a random location on the sidewalk only!! Street is not valid!! Is different from get_spawn_points()
        spawn_point.location = world.get_random_location_from_navigation()
        if spawn_point.location is not None:
            spawn_point.location.z += 2
            spawn_points.append(spawn_point)
    # 2. Choose walker and spwawn point
    batch = []
    for sp in spawn_points:
        walker_bp = random.choice(bp_pedestrians)
        batch.append(carla.command.SpawnActor(walker_bp,sp))
    # 3. Apply batch
    walker_list = []
    results= client.apply_batch_sync(batch,True)
    for res in results:
        if res.error:
            print(res.error)
        else:
            walker_list.append({"id": res.actor_id})
    batch = []
    # 4. Set walker controller
    walker_controller = world.get_blueprint_library().find('controller.ai.walker')
    for walker in walker_list:
        batch.append(carla.command.SpawnActor(walker_controller, carla.Transform(), walker["id"]))
    # 5. Apply batch
    results = client.apply_batch_sync(batch,True)
    for i in range(len(results)):
        if results[i].error:
            print(results[i].error)
        else:
            walker_list[i]["con"] = results[i].actor_id
    # 6. set all together
    all_id = []
    for walker in walker_list:
        all_id.append(walker["con"])
        all_id.append(walker["id"])
    all_actors = world.get_actors(all_id)

    world.tick()


    for i in range(0, len(all_actors), 2):
        all_actors[i].start()
        all_actors[i].go_to_location(world.get_random_location_from_navigation())

    return all_actors

def attach_collision_sensor(args, world, client, ego_vehicle):
    """Attach a collision sensor to the ego vehicle.
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
        ego_vehicle (carla.Vehicle): The ego vehicle object.
    Returns:
        sensor (carla.Sensor): The collision sensor object.
    """
    # Get the blueprint library
    collision_bp = world.get_blueprint_library().find("sensor.other.collision")
    #This sensor does not have any attributes to set, so we can just spawn it
    pos = carla.Transform()
    sensor = world.spawn_actor(collision_bp, pos, attach_to=ego_vehicle)
    return sensor

def attach_rbgcamera(args, world,client,ego_vehicle):
    """Attach a camera to the ego vehicle.
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
        ego_vehicle (carla.Vehicle): The ego vehicle object.
    Returns:
        sensor (carla.Sensor): The camera sensor object.
    """
    # Get the blueprint library
    rgb_bp: carla.ActorBlueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", args.image_size_x)
    rgb_bp.set_attribute("image_size_y", args.image_size_y)
    rgb_bp.set_attribute("fov", '90')
    rgb_bp.set_attribute("sensor_tick", args.sensor_tick)


    #rgb_bp.set_attribute("post_process_profile", "Town10HD_Opt")
    #rgb_bp.set_attribute("fstop", "9.8000001907348633")
    #rgb_bp.set_attribute("min_fstop", "1.2")
    #rgb_bp.set_attribute("iso", "300000")
    #rgb_bp.set_attribute("gamma", "1.2")
    #rgb_bp.set_attribute("temp", "7700.0")
    #rgb_bp.set_attribute("toe", "0.55")
    #rgb_bp.set_attribute("slope", "0.88")
    #rgb_bp.set_attribute("tint", "-0.15")
    #rgb_bp.set_attribute("lens_flare_intensity", "1")
    #rgb_bp.set_attribute("shutter_speed", "15.0")
    #rgb_bp.set_attribute("bloom_intensity", "0.675")
    #rgb_bp.set_attribute("focal_distance", "250")
    #rgb_bp.set_attribute("exposure_compensation", "0")
    #rgb_bp.set_attribute("exposure_min_bright", "-10")
    #rgb_bp.set_attribute("exposure_max_bright", "20")
    #rgb_bp.set_attribute("exposure_speed_up", "3")
    #rgb_bp.set_attribute("exposure_speed_down", "1")
    #rgb_bp.set_attribute("vignette_intensity", "0.7")
    #rgb_bp.set_attribute("motion_blur_intensity", "0.5")
    #rgb_bp.set_attribute("chromatic_aberration_intensity", "0.0")
    # Camera has also another attribute called "sensor_tick" that is the time to capture frames. Since we are in sync mode with a fixed time = 0.05, each tick() of 0.05 will
    # caputre a frame. If sensore_tick is set to 0.1, the camera will capture a frame every 2 ticks.
    pos = carla.Transform()
    # For the Mustang, the camera is located on the front windshield
    pos.location.z = 1.3
    pos.location.x = 0.2
    sensor = world.spawn_actor(rgb_bp, pos, attach_to=ego_vehicle)
    return sensor

def attach_lidar(args,world,client,ego_vehicle):
    """Attach a LIDAR sensor to the ego vehicle.
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
        ego_vehicle (carla.Vehicle): The ego vehicle object.
    Returns:
        sensor (carla.Sensor): The LIDAR sensor object.
    """
    # Get the blueprint library
    lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")

    # Meters of range
    lidar_bp.set_attribute("range", '100')
    # Rotation frequency should be equal to FPS in order to do a 360 ° rotation
    #NO NOISE
    lidar_bp.set_attribute('dropoff_general_rate', '0.0')
    lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
    lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
    #NO NOISE
    lidar_bp.set_attribute("rotation_frequency", str(args.fps))
    # Number of lasers
    lidar_bp.set_attribute("channels", '64')
    lidar_bp.set_attribute("upper_fov", '2')
    lidar_bp.set_attribute("lower_fov", '-24.8')
    lidar_bp.set_attribute("points_per_second", '2500000')
    lidar_bp.set_attribute("sensor_tick", args.sensor_tick)

    # Camera has also another attribute called "sensor_tick" that is the time to capture frames. Since we are in sync mode with a fixed time = 0.05, each tick() of 0.05 will
    # caputre a frame. If sensore_tick is set to 0.1, the camera will capture a frame every 2 ticks.
    pos = carla.Transform()
    #For the Mustang, the lidar is located on the roof
    pos.location.z = 1.9
    pos.location.x = -0.4
    sensor = world.spawn_actor(lidar_bp, pos, attach_to=ego_vehicle)
    return sensor

def attach_radar(args, world,client,ego_vehicle):
    """Attach a radar sensor to the ego vehicle.
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
        ego_vehicle (carla.Vehicle): The ego vehicle object.
    Returns:
        sensor (carla.Sensor): The radar sensor object.
    """
    # Get the blueprint library
    radar_bp = world.get_blueprint_library().find("sensor.other.radar")
    radar_bp.set_attribute("horizontal_fov", '35')
    radar_bp.set_attribute("vertical_fov", '20')
    radar_bp.set_attribute("points_per_second", '10000')
    radar_bp.set_attribute("range", '50')
    radar_bp.set_attribute("sensor_tick", args.sensor_tick)
    pos = carla.Transform()
    # FOr the Mustang, the radar is located at the front plate
    pos.location.z = 0.3
    pos.location.x = 2.28
    sensor = world.spawn_actor(radar_bp, pos, attach_to=ego_vehicle)
    return sensor

def attach_semantic_lidar(args, world,client,ego_vehicle):
    """Attach a semantic LIDAR sensor to the ego vehicle.
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
        ego_vehicle (carla.Vehicle): The ego vehicle object.
    Returns:
        sensor (carla.Sensor): The semantic LIDAR sensor object.
    """
    # Get the blueprint library
    semantic_lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
    # Meters of range
    semantic_lidar_bp.set_attribute("range", '100')
    # Rotation frequency should be equal to FPS in order to do a 360 ° rotation

    semantic_lidar_bp.set_attribute("rotation_frequency", str(args.fps))
    # Number of lasers
    semantic_lidar_bp.set_attribute("channels", '64')
    semantic_lidar_bp.set_attribute("upper_fov", '5')
    semantic_lidar_bp.set_attribute("lower_fov", '-25')
    semantic_lidar_bp.set_attribute("points_per_second", '3500000')
    semantic_lidar_bp.set_attribute("sensor_tick", args.sensor_tick)
    pos = carla.Transform()
    pos.location.z = 1.9
    pos.location.x = -0.4
    sensor = world.spawn_actor(semantic_lidar_bp, pos, attach_to=ego_vehicle)
    return sensor

def attach_semantic_camera(args, world,client,ego_vehicle):
    """Attach a semantic camera to the ego vehicle.
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
        ego_vehicle (carla.Vehicle): The ego vehicle object.
    Returns:
        sensor (carla.Sensor): The semantic camera sensor object.
    """
    # Get the blueprint library
    semantic_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
    semantic_bp.set_attribute("image_size_x", args.image_size_x)
    semantic_bp.set_attribute("image_size_y", args.image_size_y)
    semantic_bp.set_attribute("fov", '90')
    semantic_bp.set_attribute("sensor_tick", args.sensor_tick)
    pos = carla.Transform()
    pos.location.z = 1.3
    pos.location.x = 0.2
    sensor = world.spawn_actor(semantic_bp, pos, attach_to=ego_vehicle)
    return sensor

def attach_instance_camera(args, world,client,ego_vehicle):
    """Attach an instance camera to the ego vehicle.
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
        ego_vehicle (carla.Vehicle): The ego vehicle object.
    Returns:
        sensor (carla.Sensor): The instance camera sensor object.
    """
    # Get the blueprint library
    instance_bp = world.get_blueprint_library().find("sensor.camera.instance_segmentation")
    instance_bp.set_attribute("image_size_x", args.image_size_x)
    instance_bp.set_attribute("image_size_y", args.image_size_y)
    instance_bp.set_attribute("fov", '90')
    instance_bp.set_attribute("sensor_tick", args.sensor_tick)
    pos = carla.Transform()
    pos.location.z = 1.3
    pos.location.x = 0.2
    sensor = world.spawn_actor(instance_bp, pos, attach_to=ego_vehicle)
    return sensor

def attach_depth_camera(args, world,client,ego_vehicle):
    """Attach a depth camera to the ego vehicle.
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
        ego_vehicle (carla.Vehicle): The ego vehicle object.
    Returns:
        sensor (carla.Sensor): The depth camera sensor object.
    """
    # Get the blueprint library
    depth_bp = world.get_blueprint_library().find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", args.image_size_x)
    depth_bp.set_attribute("image_size_y", args.image_size_y)
    depth_bp.set_attribute("fov", '90')
    depth_bp.set_attribute("sensor_tick", args.sensor_tick)
    pos = carla.Transform()
    pos.location.z = 1.3
    pos.location.x = 0.2
    sensor = world.spawn_actor(depth_bp, pos, attach_to=ego_vehicle)
    return sensor

def save_lidar(lidar_measurements,anomaly_name,run):
    """Save the LIDAR measurements.
    Args:
        lidar_measurements (carla.LidarMeasurement): The LIDAR measurements object.
    """
    VIRIDIS = np.array(cm.get_cmap('plasma').colors)
    VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

    data = np.copy(np.frombuffer(lidar_measurements.raw_data, dtype=np.dtype('f4')))
    points = np.reshape(data, (int(data.shape[0] / 4), 4))
    #point's shape is smnthing like (n,4)

    intensity = points[:,3]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # The points are in the form [x, y, z, intensity], we need to convert them to the form [x, y, z] and invert the y axis (this because o3d uses a right-handed coordinate system)
    tmp = points[:,:-1]
    #Invert y axis
    tmp[:,1] = -tmp[:,1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tmp)
    pcd.colors = o3d.utility.Vector3dVector(int_color)
    # crete the direcotry if it does not exist
    # if not os.path.exists("output/"+str(run)+"/lidar/"):
    #     os.makedirs("output/"+str(run)+"/lidar/")
    if not os.path.exists("output/"+str(run)+"/lidar/raw/"):
        os.makedirs("output/"+str(run)+"/lidar/raw/")
    # if anomaly_name is not None:
    #     o3d.io.write_point_cloud(f"output/"+str(run)+"/lidar/"+"/"+anomaly_name+f"_lidar-{lidar_measurements.frame}.ply", pcd)
    # else:
    #     o3d.io.write_point_cloud(f"output/"+str(run)+"/lidar/"+"/normal_" + f"_lidar-{lidar_measurements.frame}.ply", pcd)
    tmp = np.frombuffer(lidar_measurements.raw_data, dtype=np.dtype('f4')).copy().reshape(-1,4)
    tmp[:,1] = -tmp[:,1]
    np.save(f"output/"+str(run)+"/lidar/raw/"+"/"+f"lidar-{lidar_measurements.frame}.npy", tmp)

def save_semantic_lidar(lidar_measurements,anomaly_name,run):
    """Save the Semantic LIDAR measurements.
    Args:
        lidar_measurements (carla.LidarMeasurement): The LIDAR measurements object.
    """
    labels_colour = np.array([
    (0, 0, 0), # unlabeled = 0
    # cityscape
    (128, 64, 128), # road = 1
    (244, 35, 232), # sidewalk = 2
    (70, 70, 70), # bilding = 3
    (102, 102, 156), # wall = 4
    (190, 153, 153), # fence = 5
    (153, 153, 153), # pole = 6
    (250, 170, 30), # trafficlight = 7
    (220, 220, 0), # trafficsign = 8
    (107, 142, 35), # vegetation = 9
    (152, 251, 152), # terrain = 10
    (70, 130, 180), # sky = 11
    (220, 20, 60), # pedestrian = 12
    (255, 0, 0), # rider = 13
    (0, 0, 142), # Car = 14
    (0, 0, 70), # trck = 15
    (0, 60, 100), # bs = 16
    (0, 80, 100), # train = 17
    (0, 0, 230), # motorcycle = 18
    (119, 11, 32), # bicycle = 19
                                   # cstom
    (110, 190, 160), # static = 20
    (170, 120, 50), # dynamic = 21
    (55, 90, 80), # other = 22
    (45, 60, 150), # water = 23
    (157, 234, 50), # roadline = 24
    (81, 0, 81), # grond = 25
    (150, 100, 100), # bridge = 26
    (230, 150, 140), # railtrack = 27
    (180, 165, 180), # gardrail = 28
    (180, 130, 70), # rock = 29
        #anomalies
    (193, 71, 71), # Static_Anomaly = 30
    (102, 102, 255), # Dynamic_Anomaly = 31
    (175, 83, 83), # Animal = 32
    (232, 188, 188), # Tinyanomaly = 33
    (229, 137, 137), # Smallanomaly = 34
    (189, 47, 47), # Mediumanomaly = 35
    (131, 7, 7), # Largeanomaly = 36
    ]) / 255.0

    data = np.frombuffer(lidar_measurements.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

    points = np.array([data['x'], -data['y'], data['z']]).T


    # # An example of adding some noise to our data if needed:
    # points += np.random.uniform(-0.05, 0.05, size=points.shape)

    # Colorize the pointcloud based on the CityScapes color palette
    labels = np.array(data['ObjTag'])
    int_color = labels_colour[labels]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(int_color)
    # crete the direcotry if it does not exist
    # if not os.path.exists("output/"+str(run)+"/semantic_lidar/"):
    #     os.makedirs("output/"+str(run)+"/semantic_lidar/")
    if not os.path.exists("output/"+str(run)+"/semantic_lidar/raw/"):
        os.makedirs("output/"+str(run)+"/semantic_lidar/raw/")
    # if anomaly_name is not None:
    #     o3d.io.write_point_cloud(f"output/"+str(run)+"/semantic_lidar/"+"/"+anomaly_name+f"_semantic_lidar-{lidar_measurements.frame}.ply", pcd)
    # else:
    #     o3d.io.write_point_cloud(f"output/"+str(run)+"/semantic_lidar/"+"/normal_" + f"_semantic_lidar-{lidar_measurements.frame}.ply", pcd)
    data = np.frombuffer(lidar_measurements.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
    data = np.copy(data)
    data['y'] = -data['y']
    np.save(f"output/"+str(run)+"/semantic_lidar/raw/"+"/"+f"semantic_lidar-{lidar_measurements.frame}.npy", data)

def save_radar(radar_measurements,anomaly_name,run):
    """Save the radar measurements.
    Args:
        radar_measurements (carla.RadarMeasurement): The radar measurements object.
    """
    # Convert the radar measurements to a numpy array
    # Raw data is the form of [velocity, azimuth, altitude, depth]
    radar_points_list = []
    cols = []
    for detect in radar_measurements:
        x = detect.depth * math.cos(detect.altitude) * math.cos(detect.azimuth)
        y = detect.depth * math.cos(detect.altitude) * math.sin(detect.azimuth)
        z = detect.depth * math.sin(detect.altitude)

        radar_points_list.append([x,y,z,detect.velocity])

        def clamp(min_v, max_v, value):
            return max(min_v, min(value, max_v))
        norm_velocity = detect.velocity / 7.5
        r = int(clamp(0.0, 1.0, 1.0 - norm_velocity))
        g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)))
        b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)))
        cols.append([r,g,b])

    radar_points_list = np.array(radar_points_list)
    pcd = o3d.geometry.PointCloud()
    points = radar_points_list[:,:-1]
    points[:,1] = -points[:,1]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    # crete the direcotry if it does not exist
    if not os.path.exists("output/"+str(run)+"/radar/"):
        os.makedirs("output/"+str(run)+"/radar/")
    if not os.path.exists("output/"+str(run)+"/radar/raw/"):
        os.makedirs("output/"+str(run)+"/radar/raw/")
    if anomaly_name:
        o3d.io.write_point_cloud(f"output/"+str(run)+"/radar/"+"/"+anomaly_name+str(run)+f"_radar-{radar_measurements.frame}.ply", pcd)
    else:
        o3d.io.write_point_cloud(f"output/"+str(run)+"/radar/"+"/normal_" + str(run) + f"_radar-{radar_measurements.frame}.ply", pcd)
    np.save(f"output/"+str(run)+"/radar/raw/"+"/"+f"radar-{radar_measurements.frame}.npy", np.frombuffer(radar_measurements.raw_data))

def save_radar_old(radar_measurements,anomaly_name,run):
    """Save the radar measurements.
    Args:
        radar_measurements (carla.RadarMeasurement): The radar measurements object.
    """
    # Convert the radar measurements to a numpy array
    # Raw data is the form of [velocity, azimuth, altitude, depth]
    radar_points_list = []
    for detect in radar_measurements:
        x = detect.depth * math.cos(detect.altitude) * math.cos(detect.azimuth)
        y = detect.depth * math.cos(detect.altitude) * math.sin(detect.azimuth)
        z = detect.depth * math.sin(detect.altitude)

        radar_points_list.append([x,y,z,detect.velocity])
    radar_points_list = np.array(radar_points_list)
    intensity = np.abs(radar_points_list[:,-1])
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    COOL_RANGE = np.linspace(0.0, 1.0, 256)
    COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))[:, :3]

    int_color = np.c_[
        np.interp(intensity_col, COOL_RANGE, COOL[:, 0]),
        np.interp(intensity_col, COOL_RANGE, COOL[:, 1]),
        np.interp(intensity_col, COOL_RANGE, COOL[:, 2])
    ]

    pcd = o3d.geometry.PointCloud()
    points = radar_points_list[:,:-1]
    points[:,1] = -points[:,1]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(int_color)
    # crete the direcotry if it does not exist
    if not os.path.exists("output/"+str(run)+"/radar/"):
        os.makedirs("output/"+str(run)+"/radar/")
    if anomaly_name:
        o3d.io.write_point_cloud(f"output/"+str(run)+"/radar/"+"/"+anomaly_name+str(run)+f"_radar-{radar_measurements.frame}.ply", pcd)
    else:
        o3d.io.write_point_cloud(f"output/"+str(run)+"/radar/"+"/normal_" + str(run) + f"_radar-{radar_measurements.frame}.ply", pcd)

def view_lidar(lidar_name):
    """View the LIDAR measurements.
    Args:
        lidar_name (str): The name of the LIDAR file to view.
    """
    pcd = o3d.io.read_point_cloud("output/lidar/",lidar_name)
    o3d.visualization.draw_geometries([pcd])

def spawn_anomaly(world,client,ego_vehicle,prop,is_dynamic,is_character,can_be_rotated,size,anomaly_in_waypoint=True,spawn_at_zero=False,spawn_on_right=False,big_mesh=False,skip_check=False):
    """Spawn an anomaly in the CARLA simulator. The anomaly will have the same rotation of the ego vehicle
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
        ego_vehicle (carla.Vehicle): The ego vehicle object.
    """
    # Get the unit vector of the vehicle's forward direction, this is used to spawn objects in front of the vehicle

    map:carla.Map = world.get_map()
    forward_vector = ego_vehicle.get_transform().rotation.get_forward_vector()
    right_vector = ego_vehicle.get_transform().rotation.get_right_vector()
    # Now add the distance to the vehicle's location to get the new location, the distance will be based on the vehicle's forward vector

    bp_lib: carla.BlueprintLibrary = world.get_blueprint_library()
    # There's the flipped car anomaly that spawns a vehicle.*
    if prop.startswith("vehicle."):
        anomalies = bp_lib.filter(f"{prop}")
    else:
        anomalies = bp_lib.filter(f"blueprint.{prop}")
    if len(anomalies) == 0:
        print(f"Anomaly {prop} not found.")
        return None
    anomaly: carla.ActorBlueprint = anomalies[0]

    if not spawn_at_zero:
        anomaly_tmp = world.spawn_actor(anomaly, carla.Transform(carla.Location(0, 0, 60), carla.Rotation(0, 0, 0)))
        trial = 0
        while True:
            trial += 1
            if size == "tiny":
                distance = random.uniform(10,15)
            elif size == "small":
                distance = random.uniform(15,20)
            elif size == "medium":
                distance = random.uniform(20,25)
            elif size == "large":
                distance = random.uniform(25,30)
            else:
                print("Invalid size, using 10 to 30")
                distance = random.uniform(10,30)
            # The dynamic anomalies are spawned on the sidewalk, on the right of the ego vehicle, so it's fixed, meanwhile the static anomalies are spawned randomly
            if not is_dynamic:
                right_left = random.uniform(-6,6)
            else:
                # The dynamic anomalies are spawned on the right of the ego vehicle, pick a random from 4 to 6 meters
                right_left = random.uniform(3,5)
            if spawn_on_right:
                right_left = random.uniform(3,5)
            distance_vector = right_left*right_vector + distance*forward_vector
            location = ego_vehicle.get_transform().location + distance_vector
            if spawn_on_right:
                # LaneType.Shoulder is the right side of the road (like parking), LaneType.Sidewalk is the sidewalk
                wp = map.get_waypoint(location, lane_type=carla.LaneType.Shoulder)
                location.x = wp.transform.location.x
                location.y = wp.transform.location.y

            if skip_check:
                break

            extent = anomaly_tmp.bounding_box.extent
            #The extent is half the size of the bounding box, so we need to multiply by 2 to get the full size
            #Here we check if the object is colliding with something, we do this by casting rays in the area of the object
            # cast_ray return the points of intersection with the world, if the length of the points is more than 1 (just one is fine, it's the ground), it means that the ray hit something
            # so we need to find a new location.
            # step_size is the distance between each ray, the smaller the step_size, the more rays are casted, but the more computationally expensive it is


            # Let's say that we cast a ray each 0.2 meters, but if the mash is very small, we can cast a ray each 0.05 meters
            # If one between the extent.x and extent.y is less than 0.1, we set the step_size to 0.05

            if extent.x < 0.2 or extent.y < 0.2:
                step_size = 0.05
            else:
                step_size = 0.1
            # We add a small margin to the extent to avoid issues with the bounding box colliding ad edges with other bounding boxes
            extent.x += 0.05
            extent.y += 0.05
            points_x = extent.x * 2 / step_size
            points_y = extent.y * 2 / step_size
            is_valid = True
            #If we have one on this below, we are good to go. NONE means no label, so is fine too (for example a box collision has NONE label)
            valid_labels = [carla.CityObjectLabel.Roads, carla.CityObjectLabel.Sidewalks, carla.CityObjectLabel.Ground,
                            carla.CityObjectLabel.RoadLines, carla.CityObjectLabel.Terrain, carla.CityObjectLabel.NONE,
                            carla.CityObjectLabel.Water]
            for i in range(int(points_x)):
                for j in range(int(points_y)):
                    x = location.x - extent.x + i * step_size
                    y = location.y - extent.y + j * step_size
                    test_location = carla.Location(x=x,y=y,z=10)
                    points = world.cast_ray(test_location, test_location - carla.Location(z=100))
                    if len(points) != 1:
                        labels = [point.label for point in points]
                        # Check if each label in labels is in valid_labels
                        is_valid = all(x in valid_labels for x in labels)
                        if not is_valid:
                            break
                if not is_valid:
                    break
            if is_valid:
                break
            print(f"Anomaly {prop} colliding with {labels}, at {location} finding new location...")
            if trial == 20:
                anomaly_tmp.destroy()
                print(f"Tried too many times to spawn {prop} anomaly. Probably it doesn't fit. Skipping it...")
                return None

    if spawn_at_zero:
        location = carla.Location(x=0,y=0,z=0)

    rotation: carla.Rotation = ego_vehicle.get_transform().rotation
    if not is_dynamic:
        rotation.yaw = random.uniform(-180,180)
        if can_be_rotated:
            rotation.roll = random.uniform(-180,180)
            rotation.pitch = random.uniform(-180,180)

    transform:carla.Transform = carla.Transform(location,rotation)

    if is_character:
        # If is a char, 1 is ok
        transform.location.z = 1
    else:
        if not spawn_at_zero and not skip_check:
            #Otherwise compute the z coordinate based on the bounding box of the object
            transform.location.z = 10
            # We set z = 10, but is too high, so we need to adjust it to be on the ground
            # We use the bounding box to get the lowest point of the object, and we use the ground projection to get the ground level
            # Then we adjust the z coordinate of the object

            # Get the ground level of our location, from this we will get the z coordinate
            valid_labels = [carla.CityObjectLabel.Roads, carla.CityObjectLabel.Sidewalks, carla.CityObjectLabel.Ground,
                            carla.CityObjectLabel.RoadLines, carla.CityObjectLabel.Terrain]
            points = world.cast_ray(transform.location, transform.location - carla.Location(z=15))
            #filter the point, keeping only the ones that are in valid_labels
            points = [point for point in points if point.label in valid_labels]
            if len(points) != 0:
                street = points[-1]
                ground_z = street.location.z
            else:
                ground_z = 0
            if ground_z < 0:
                ground_z = 0
            # Get the verteces of the bounding box and get the lowest z coordinate
            bb = anomaly_tmp.bounding_box
            vertices = bb.get_world_vertices(transform)
            min_z = min([vertex.z for vertex in vertices])
            max_z = max([vertex.z for vertex in vertices])
            # Calculate the difference between the lowest point of the object and the ground level
            diff = min_z - ground_z
            # Adjust the z coordinate of the object
            transform.location.z = transform.location.z - diff + 0.8

    if anomaly_in_waypoint:
        wp = map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Sidewalk)
        transform.location.x = wp.transform.location.x
        transform.location.y = wp.transform.location.y

    #We destroy the test anomaly and spawn the real one, with the right transform
    if not spawn_at_zero:
        anomaly_tmp.destroy()
        world.tick()
    anomaly_actor = world.try_spawn_actor(anomaly, transform)


    world.tick()

    if anomaly_actor is None:
        print(f"Failed to spawn {prop} anomaly.")
        return None
    print(f"Spawned {prop} anomaly.")
    return anomaly_actor

def obb_corners(origin, extent):
    ex = extent.x
    ey = extent.y
    ez = extent.z

    ret = [
        origin + carla.Location(ex, ey, ez),
        origin + carla.Location(-ex, ey, ez),
        origin + carla.Location(-ex, -ey, ez),
        origin + carla.Location(ex, -ey, ez),
        origin + carla.Location(ex, ey, -ez),
        origin + carla.Location(-ex, ey, -ez),
        origin + carla.Location(-ex, -ey, -ez),
        origin + carla.Location(ex, -ey, -ez),
    ]

    return  ret # world-space corners

def center_vertices_world(origin, vertices):
    # returns vertices translated so origin becomes (0,0,0)
    return vertices - np.array(origin)

def euler_to_rotmat(roll_deg, pitch_deg, yaw_deg):
    # Unreal FRotator: (Roll, Pitch, Yaw) in degrees (X,Y,Z)
    r = radians(roll_deg); p = radians(pitch_deg); y = radians(yaw_deg)
    Rx = np.array([[1,0,0],[0,cos(r),-sin(r)],[0,sin(r),cos(r)]])
    Ry = np.array([[cos(p),0,sin(p)],[0,1,0],[-sin(p),0,cos(p)]])
    Rz = np.array([[cos(y),-sin(y),0],[sin(y),cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx   # order: roll then pitch then yaw

def center_and_unrotate(origin, vertices, rot_euler_deg):
    # Translate to origin, then rotate by inverse rotation to get box-local coords
    trans = center_vertices_world(origin, vertices)
    R = euler_to_rotmat(*rot_euler_deg)   # rot from local->world
    Rinv = R.T                            # inverse of rotation matrix
    return (Rinv @ trans.T).T

def destroy_anomalies(world,client,anomaly_actors):
    """Destroy the anomalies in the CARLA simulator.
    Args:
        world (carla.World): The CARLA world object.
        client (carla.Client): The CARLA client object.
        anomaly_actors (list): List of anomaly actors to destroy.
    """
    print("Destroying anomalies...")
    client.apply_batch([carla.command.DestroyActor(x) for x in anomaly_actors])
    print("Destroying anoamlies... Done.")