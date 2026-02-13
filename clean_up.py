from utils import destroy_all_vehicles, destroy_all_pedestrians, set_async_mode
import carla

client = carla.Client('localhost', 2000)
world:carla.World = client.get_world()
set_async_mode(world,client)

destroy_all_vehicles(world,client)
destroy_all_pedestrians(world,client)

client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter("sensor.*")])
client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter("blueprint.*")])
