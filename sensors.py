import carla
from utils import *

class Sensor:
    """
    A class representing a sensor. This has a method called 'handle' that handles the sensor data, that will be overridden by subclasses.
    """
    def __init__(self, queue, sensor):
        self.queue = queue
        self.sensor:carla.Sensor = sensor

    def handle(self, data, anomaly):
        """
        Handle the sensor data. This method should be overridden by subclasses.
        :param sensor_tick:
        :param data: The data to handle.
        :param anomaly: The anomaly name.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def destroy(self):
        """
        Destroy the sensor.
        """
        if self.sensor.is_listening():
            self.sensor.stop()
            self.sensor.destroy()

    def get_anomaly_name_from_list(self, anomaly_list):
        """
        Get the anomaly name from a list of anomalies.
        :param anomaly_list:
        :return:
        """
        anomaly_name = ""
        if anomaly_list:
            for anomaly in anomaly_list:
                anomaly_name = anomaly_name + anomaly + "_"
            return anomaly_name
        return None

class RGB_Sensor(Sensor):
    """
    A class representing an RGB sensor. This inherits from the Sensor class and implements the handle method.
    """

    def __init__(self, queue, sensor):
        """
        Initialize the RGB sensor.
        :param queue: The queue to store sensor data.
        :param sensor: The CARLA sensor object.
        """
        super().__init__(queue, sensor)

    def handle(self, run, anomaly=None):
        """
        Handle the RGB sensor data.
        :param sensor_tick:
        :param run: The current run number.
        :param anomaly: The anomaly name.
        """
        # Process the RGB data here
        anomaly_name = self.get_anomaly_name_from_list(anomaly)
        # if len(self.queue) != 0:
        cam_data: carla.Image = self.queue.get(True,1)
        if anomaly_name:
            cam_data.save_to_disk('output/' + str(run) + '/rgb' +  "/" + anomaly_name  + '_%06d' % cam_data.frame)
        else:
            cam_data.save_to_disk('output/' + str(run) + '/rgb' +  '/normal_' + '_%06d' % cam_data.frame)

class Lidar_Sensor(Sensor):
    """
    A class representing a LiDAR sensor. This inherits from the Sensor class and implements the handle method.
    """

    def __init__(self, queue, sensor):
        """
        Initialize the LiDAR sensor.
        :param queue: The queue to store sensor data.
        :param sensor: The CARLA sensor object.
        """
        super().__init__(queue, sensor)

    def handle(self, run, anomaly=None):
        """
        Handle the LiDAR sensor data.
        :param run: The current run number.
        :param anomaly: The anomaly name.
        """
        # Process the LiDAR data here
        anomaly_name = self.get_anomaly_name_from_list(anomaly)
#        if len(self.queue) != 0:
        lidar_data: carla.LidarMeasurement = self.queue.get(True,1)
        save_lidar(lidar_data, anomaly_name, run)

class Semantic_Sensor(Sensor):
    """
    A class representing a semantic sensor. This inherits from the Sensor class and implements the handle method.
    """

    def __init__(self, queue, sensor):
        """
        Initialize the semantic sensor.
        :param queue: The queue to store sensor data.
        :param sensor: The CARLA sensor object.
        """
        super().__init__(queue, sensor)

    def handle(self, run, anomaly=None):
        """
        Handle the semantic sensor data.
        :param run: The current run number.
        :param anomaly: The anomaly name.
        """
        # Process the semantic data here
        anomaly_name = self.get_anomaly_name_from_list(anomaly)
        # if len(self.queue) != 0:
        semantic_data: carla.Image = self.queue.get(True,1)
        if anomaly_name:
            semantic_data.save_to_disk(
                'output/' + str(run) +'/semantic/original' + "/" + anomaly_name +  '_%06d' % semantic_data.frame)
            semantic_data.save_to_disk(
                f'output/' + str(run) +'/semantic/converted' + "/" + anomaly_name + f'_{semantic_data.frame}',
                carla.ColorConverter.CityScapesPalette)
        else:
            semantic_data.save_to_disk(
                'output/' + str(run) +'/semantic/original'  + '/normal_'  + '_%06d' % semantic_data.frame)
            semantic_data.save_to_disk(f'output/' + str(run) +'/semantic/converted' + '/normal_'  + f'_{semantic_data.frame}',
                                       carla.ColorConverter.CityScapesPalette)

class Semantic_Lidar_Sensor(Sensor):
    """
    A class representing a semantic LiDAR sensor. This inherits from the Sensor class and implements the handle method.
    """

    def __init__(self, queue, sensor):
        """
        Initialize the semantic LiDAR sensor.
        :param queue: The queue to store sensor data.
        :param sensor: The CARLA sensor object.
        """
        super().__init__(queue, sensor)

    def handle(self, run, anomaly=None):
        """
        Handle the semantic LiDAR sensor data.
        :param run: The current run number.
        :param anomaly: The anomaly name.
        """
        # Process the semantic LiDAR data here
        anomaly_name = self.get_anomaly_name_from_list(anomaly)
        # if len(self.queue) != 0:
        lidar_semantic_data: carla.SemanticLidarMeasurement = self.queue.get(True,1)
        save_semantic_lidar(lidar_semantic_data, anomaly_name, run)

class Radar_Sensor(Sensor):
    """
    A class representing a radar sensor. This inherits from the Sensor class and implements the handle method.
    """

    def __init__(self, queue, sensor):
        """
        Initialize the radar sensor.
        :param queue: The queue to store sensor data.
        :param sensor: The CARLA sensor object.
        """
        super().__init__(queue, sensor)

    def handle(self, run, anomaly=None):
        """
        Handle the radar sensor data.
        :param run: The current run number.
        :param anomaly: The anomaly name.
        """
        # Process the radar data here
        anomaly_name = self.get_anomaly_name_from_list(anomaly)
        # if len(self.queue) != 0:
        radar_data: carla.RadarMeasurement = self.queue.get(True,1)
        save_radar(radar_data, anomaly_name, run)

class Depth_Sensor(Sensor):
    """
    A class representing a depth sensor. This inherits from the Sensor class and implements the handle method.
    """

    def __init__(self, queue, sensor):
        """
        Initialize the depth sensor.
        :param queue: The queue to store sensor data.
        :param sensor: The CARLA sensor object.
        """
        super().__init__(queue, sensor)

    def handle(self, run, anomaly=None):
        """
        Handle the depth sensor data.
        :param run: The current run number.
        :param anomaly: The anomaly name.
        """
        # Process the depth data here
        anomaly_name = self.get_anomaly_name_from_list(anomaly)
        # if len(self.queue) != 0:
        depth_data: carla.Image = self.queue.get(True,1)
        if anomaly_name:
            depth_data.save_to_disk(
                'output/' + str(run) +'/depth/original'  + "/" + anomaly_name  + '_%06d' % depth_data.frame)
            depth_data.save_to_disk(
                'output/' + str(run) +'/depth/depth'  + "/" + anomaly_name  + '_%06d' % depth_data.frame,
                carla.ColorConverter.Depth)
            depth_data.save_to_disk(
                'output/' + str(run) +'/depth/logaritmic'  + "/" + anomaly_name  + '_%06d' % depth_data.frame,
                carla.ColorConverter.LogarithmicDepth)
        else:
            depth_data.save_to_disk('output/' + str(run) +'/depth/original'  + '/normal_' + '_%06d' % depth_data.frame)
            depth_data.save_to_disk('output/' + str(run) +'/depth/depth'  + '/normal_' + '_%06d' % depth_data.frame,
                                    carla.ColorConverter.Depth)
            depth_data.save_to_disk('output/' + str(run) +'/depth/logaritmic'  + '/normal_'  + '_%06d' % depth_data.frame,
                                    carla.ColorConverter.LogarithmicDepth)

class Instant_Segmentation_Sensor(Sensor):
    """
    A class representing an instance segmentation sensor. This inherits from the Sensor class and implements the handle method.
    """

    def __init__(self, queue, sensor):
        """
        Initialize the instance segmentation sensor.
        :param queue: The queue to store sensor data.
        :param sensor: The CARLA sensor object.
        """
        super().__init__(queue, sensor)

    def handle(self, run, anomaly=None):
        """
        Handle the instance segmentation sensor data.
        :param run: The current run number.
        :param anomaly: The anomaly name.
        """
        # Process the instance segmentation data here
        anomaly_name = self.get_anomaly_name_from_list(anomaly)
        # if len(self.queue) != 0:
        instance_data: carla.Image = self.queue.get(True,1)
        if anomaly_name:
            instance_data.save_to_disk(
                'output/' + str(run) +'/instance'  + "/" +anomaly_name  + '_%06d' % instance_data.frame)
        else:
            instance_data.save_to_disk('output/' + str(run) +'/instance'  + '/normal_'  + '_%06d' % instance_data.frame)

class Collision_Sensor(Sensor):
    """
    A class representing a collision sensor. This inherits from the Sensor class and implements the handle method.
    """

    def __init__(self, queue, sensor):
        """
        Initialize the collision sensor.
        :param queue: The queue to store sensor data.
        :param sensor: The CARLA sensor object.
        """
        super().__init__(queue, sensor)

    def handle(self, data, anomaly=None):
        """
        Handle the collision sensor data.
        :param data: The collision data to handle.
        :param anomaly: The anomaly name.
        """
        # Process the collision data here
        pass