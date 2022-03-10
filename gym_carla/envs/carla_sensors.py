import cv2
import numpy as np
import carla

class SensorManager:
    """This class instantiates sensors and processes raw sensor data"""
    def __init__(self, world, im_height, im_width):
        self.world                = world
        self.im_height            = im_height
        self.im_width             = im_width
        self.reset()
    
    def reset(self):
        """Resets all sensor output variables"""
        self.collision_hist       = []
        self.rgb_image            = np.zeros([self.im_height,self.im_width,3])
        self.semantic_image       = np.zeros([self.im_height,self.im_width,3])
        self.mask                 = np.zeros([self.im_height,self.im_width])
        self.lidar_image          = np.zeros([self.im_height,self.im_width])
        self.sem_lidar_image      = np.zeros([self.im_height,self.im_width])
        self.radar_dist           = None
        self.radar_vel            = None
    
    ####################################################################################
    #                               Return methods
    ####################################################################################

    def _get_observations(self):
        """Returns sensing ouputs"""
        # Return all sensor outputs
        return self.semantic_image, self.lidar_image
    
    def _check_for_collision(self):
        """Checks if collision occured"""
        return True if len(self.collision_hist)>0 else False
    
    def _get_nearest_radar_value(self):
        """Returns distance to nearest radar detection and its relative velocity"""
        return self.radar_dist, self.radar_vel
    
    ####################################################################################
    #                            Sensor initialization
    ####################################################################################

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        """Instantiate vehicle sensors"""

        # RGB sensor
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            # Set attributes
            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])
            # Spawn actor, set listener, add to actor list
            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.process_rgb_image)
            return camera
        
        # Semantic RGB sensor
        elif sensor_type == 'SemanticRGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            # Set attributes
            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])
            # Spawn actor, set listener, add to actor list
            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.process_semantic_rgb_image)
            return camera
        
        # Semantic RGB sensor
        elif sensor_type == 'RoadMask':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            # Set attributes
            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])
            # Spawn actor, set listener, add to actor list
            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.process_semantic_road_mask)
            return camera
        
        # LiDAR sensor
        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            # Set attributes
            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])
            # Spawn actor, set listener, add to actor list
            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)
            lidar.listen(self.process_lidar_image)
            return lidar
        
        # Semantic LiDAR sensor
        elif sensor_type == 'SemanticLiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '100')
            # Set attributes
            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])
            # Spawn actor, set listener, add to actor list
            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)
            lidar.listen(self.process_semanticlidar_image)
            return lidar
        
        # Radar sensor
        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            # Set attributes
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])
            # Spawn actor, set listener, add to actor list
            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(self.process_radar_image)
            return radar
        
        # Collision sensor
        elif sensor_type == "Collision":
            colsensor = self.world.get_blueprint_library().find("sensor.other.collision")
            colsensor = self.world.spawn_actor(colsensor, transform, attach_to=attached)
            colsensor.listen(lambda event: self.collision_data(event))
            return colsensor
        
        else:
            return None
    
    ####################################################################################
    #                            Data processing methods
    ####################################################################################
    
    def _get_road_highlights(self, where):
        """Returns True if any non-road pixel in proximity.
        Where = (-1, 0, 1) ==> (Left, Front, Right) proximity to obstacle
        """
        
        image = self.mask
        new_mask = np.zeros(image.shape)      

        if (where == -1):
            check = image[-1, :(image.shape[1]//3):] == 0
            new_mask[-1, :(image.shape[1]//3):] = image[-1, :(image.shape[1]//3):]
            if check.any():
                return True
                    
        elif (where == 0):
            check = image[(image.shape[0]//3)*2, image.shape[1]//3:(image.shape[1]//3)*2:] == 0
            new_mask[(image.shape[0]//3)*2, image.shape[1]//3:(image.shape[1]//3)*2:] = image[(image.shape[0]//3)*2, image.shape[1]//3:(image.shape[1]//3)*2:]
            if check.any():
                return True
        
        elif (where == 1):
            check = image[-1, (image.shape[1]//3)*2::] == 0
            new_mask[-1, (image.shape[1]//3)*2::] = image[-1, (image.shape[1]//3)*2::]
            if check.any():
                return True
        
        return False
    
    ####################################################################################
    #                            Sensor callback methods
    ####################################################################################
    
    def process_rgb_image(self, data):
        """Process RGB image data"""
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.im_height, self.im_width, 4))
        array = array[:,:,:3]        # 3 channel image
        array = array[:,:,::-1]
        self.rgb_image = array #cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    
    def process_semantic_rgb_image(self, data):
        """Process semantic image data and mask"""

        # Create 3 channel semantic image
        data.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.im_height, self.im_width, 4))
        array = array[:,:,:3]        # 3 channel image
        self.semantic_image = array
    
    def process_semantic_road_mask(self, data):
        """Process semantic image data to produce single channel road mask"""

        # Create single channel road mask image
        i = np.array(data.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))  # RGBA
        i2 = i2[int(self.im_height//2.4)::] # trim the top part
        sem_img = i2[:, :, 2]
        sem_img[sem_img == 6] = 255
        sem_img[sem_img == 7] = 255
        sem_img[sem_img < 255] = 0
        self.mask = sem_img
    
    def process_lidar_image(self, data):
        """Process LIDAR image data"""
        disp_size   = [self.im_width,self.im_height]
        points      = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points      = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data  = np.array(points[:, :2])
        lidar_data *= min(disp_size) / (2*100)
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data  = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data  = lidar_data.astype(np.int32)
        lidar_data  = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img   = np.zeros((lidar_img_size), dtype=np.uint8)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        self.lidar_image = np.swapaxes(lidar_img[:,:,0],0,1)
    
    def process_semanticlidar_image(self, data):
        """Process Semantic LIDAR image data"""
        disp_size   = [self.im_width,self.im_height]
        points      = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points      = np.reshape(points, (int(points.shape[0] / 6), 6))
        lidar_data  = np.array(points[:, :2])
        lidar_data *= min(disp_size) / (2*100)
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data  = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data  = lidar_data.astype(np.int32)
        lidar_data  = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img   = np.zeros((lidar_img_size), dtype=np.uint8)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        self.sem_lidar_image = np.swapaxes(lidar_img[:,:,0],0,1)
    
    def process_radar_image(self, radar_data):
        """Get radar data and find the closest detection"""
        points    = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points    = np.reshape(points, (len(radar_data), 4))
        min_dist  = np.min(points[:,3])
        min_vel   = points[np.where(points[:,3] == min_dist)[0][0],0]
        # print("FR; Dist: "+str(min_dist)+"m, relative vel: "+str(min_vel)+"m/s")

        self.radar_dist = min_dist
        self.radar_vel = min_vel
    
    def collision_data(self, event):
        """Process collision event data"""
        self.collision_hist.append(event)
