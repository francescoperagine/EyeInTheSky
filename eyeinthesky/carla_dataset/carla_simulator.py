import carla
import random
import time
import math
from typing import Tuple, List
from loguru import logger
import os
from pathlib import Path

class CarlaSimulator:
    # Class attributes
    WEATHER_OPTIONS = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.WetSunset,
        carla.WeatherParameters.ClearNight,
        carla.WeatherParameters.CloudyNight,
        carla.WeatherParameters.WetNight
    ]
    
    COLLISION_TYPES = ['rear_end', 't_bone', 'side_swipe']
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    
    def __init__(self, output_dir: str = None):
        self.client = None
        self.world = None
        self.logger = logger
        
        # Set output directory to project root by default
        if output_dir is None:
            self.output_dir = str(Path(__file__).parent.parent.parent / 'dataset')
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.connect()

    def save_image(self, image, scene_id: int, camera_id: int):
        """Save captured image to the correct directory"""
        output_path = os.path.join(self.output_dir, f'scene_{scene_id}_cam_{camera_id}.png')
        image.save_to_disk(output_path)
    
    def connect(self):
        """Connect to CARLA server"""
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
        except Exception as e:
            self.logger.error(f"Failed to connect to CARLA: {e}")
            raise

    def set_weather(self, weather):
        """Set weather in the simulation"""
        self.world.set_weather(weather)

    def get_camera_transforms(self, vehicle1: carla.Vehicle, vehicle2: carla.Vehicle) -> List[carla.Transform]:
        """Get camera positions around the collision"""
        # Calculate center point between vehicles
        v1_loc = vehicle1.get_location()
        v2_loc = vehicle2.get_location()
        center = carla.Location(
            x=(v1_loc.x + v2_loc.x) / 2,
            y=(v1_loc.y + v2_loc.y) / 2,
            z=(v1_loc.z + v2_loc.z) / 2 + 2.0  # Raise camera height
        )
        
        transforms = []
        angles = [0, 90, 180, 270]
        radius = 5.0
        
        for angle in angles:
            rad = math.radians(angle)
            camera_location = carla.Location(
                x=center.x + radius * math.cos(rad),
                y=center.y + radius * math.sin(rad),
                z=center.z
            )
            
            # Calculate rotation to look at center
            direction = center - camera_location
            camera_rotation = carla.Rotation(
                pitch=math.degrees(math.atan2(direction.z, math.sqrt(direction.x**2 + direction.y**2))),
                yaw=math.degrees(math.atan2(direction.y, direction.x)),
                roll=0.0
            )
            
            transforms.append(carla.Transform(camera_location, camera_rotation))
        
        return transforms

    def setup_collision(self, collision_type: str) -> Tuple[carla.Vehicle, carla.Vehicle]:
        """Setup vehicles for more dramatic collisions"""
        vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
        spawn_points = self.world.get_map().get_spawn_points()
        
        for v1_sp in spawn_points:
            try:
                # Spawn first vehicle
                vehicle1 = self.world.spawn_actor(vehicle_bp, v1_sp)
                
                # Setup second vehicle with more dramatic positioning
                if collision_type == 'rear_end':
                    # Place second vehicle further back with higher speed
                    v2_transform = carla.Transform(
                        v1_sp.location + carla.Location(x=-15.0),  # Further back
                        v1_sp.rotation
                    )
                    
                elif collision_type == 't_bone':
                    # Place second vehicle further to the side with higher speed
                    v2_transform = carla.Transform(
                        v1_sp.location + carla.Location(y=-15.0),
                        carla.Rotation(v1_sp.rotation.pitch, v1_sp.rotation.yaw + 90, v1_sp.rotation.roll)
                    )
                    
                else:  # side_swipe
                    # Place second vehicle at an angle for more dramatic side swipe
                    v2_transform = carla.Transform(
                        v1_sp.location + carla.Location(y=-4.0, x=-2.0),
                        carla.Rotation(v1_sp.rotation.pitch, v1_sp.rotation.yaw + 15, v1_sp.rotation.roll)
                    )
                
                # Spawn second vehicle
                vehicle2 = self.world.spawn_actor(vehicle_bp, v2_transform)
                
                # Enable physics
                vehicle1.set_simulate_physics(True)
                vehicle2.set_simulate_physics(True)
                
                # Set more dramatic velocities
                if collision_type == 'rear_end':
                    vehicle1.set_target_velocity(carla.Vector3D(5.0, 0, 0))  # Slow moving
                    vehicle2.set_target_velocity(carla.Vector3D(80.0, 0, 0))  # Fast approaching
                    
                elif collision_type == 't_bone':
                    vehicle1.set_target_velocity(carla.Vector3D(40.0, 0, 0))  # Moving target
                    vehicle2.set_target_velocity(carla.Vector3D(0, 90.0, 0))  # Fast impact
                    
                else:  # side_swipe
                    vehicle1.set_target_velocity(carla.Vector3D(50.0, 0, 0))
                    vehicle2.set_target_velocity(carla.Vector3D(60.0, 15.0, 0))  # Angled approach
                
                # Apply initial impulse for more dramatic effect
                vehicle1.add_impulse(carla.Vector3D(0, 0, -20000))  # Push down for better traction
                vehicle2.add_impulse(carla.Vector3D(0, 0, -20000))
                
                return vehicle1, vehicle2
                
            except RuntimeError as e:
                if 'vehicle1' in locals():
                    try:
                        vehicle1.destroy()
                    except:
                        pass
                continue
        
        raise RuntimeError("Could not find suitable spawn points for vehicles")

    def generate_scene(self, scene_id: int):
        """Generate a single collision scene with more dramatic timing"""
        cameras = []
        vehicles = []
        
        try:
            # Set weather
            weather = random.choice(self.WEATHER_OPTIONS)
            self.set_weather(weather)
            
            # Select collision type (was missing before)
            collision_type = random.choice(self.COLLISION_TYPES)
            
            self.logger.info(f"Generating scene {scene_id} - Weather: {weather}, Collision: {collision_type}")
            
            # Setup vehicles with the selected collision type
            vehicle1, vehicle2 = self.setup_collision(collision_type)
            vehicles.extend([vehicle1, vehicle2])
            
            # Setup cameras
            camera_transforms = self.get_camera_transforms(vehicle1, vehicle2)
            for i, transform in enumerate(camera_transforms):
                camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
                camera_bp.set_attribute('image_size_x', str(self.IMAGE_WIDTH))
                camera_bp.set_attribute('image_size_y', str(self.IMAGE_HEIGHT))
                camera_bp.set_attribute('fov', '90')
                
                camera = self.world.spawn_actor(camera_bp, transform)
                cameras.append((i, camera))
                
                camera.listen(lambda image, cam_id=i: self.save_image(image, scene_id, cam_id))
            
            # Fast forward with more ticks and dynamic camera captures
            for tick in range(100):  # More ticks for collision development
                self.world.tick()
                
                if tick > 80:  # Capture images near collision point
                    for _, camera in cameras:
                        # Force camera update
                        self.world.tick()
            
        except Exception as e:
            self.logger.error(f"Error generating scene: {e}")
        finally:
            # Clean up
            for vehicle in vehicles:
                try:
                    if vehicle is not None:
                        vehicle.destroy()
                except:
                    pass
            
            for _, camera in cameras:
                try:
                    if camera is not None:
                        camera.stop()
                        camera.destroy()
                except:
                    pass

    def generate_dataset(self, num_scenes: int = 10):
        """Generate multiple collision scenes"""
        self.logger.info(f"Starting dataset generation: {num_scenes} scenes")
        
        for scene_id in range(num_scenes):
            self.generate_scene(scene_id)
            
        self.logger.info("Dataset generation completed")

def main():
    # Specify output directory at project root
    project_root = Path(__file__).parent.parent.parent
    output_dir = str(project_root / 'dataset')
    
    simulator = CarlaSimulator(output_dir=output_dir)
    simulator.generate_dataset(num_scenes=1)

if __name__ == "__main__":
    main()