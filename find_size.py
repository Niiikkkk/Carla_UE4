import carla

def find_size_anomaly():
    list_static_anomalies = [

    ]

    client: carla.Client = carla.Client("localhost", 2000)
    world: carla.World = client.get_world()
    bp_lib = world.get_blueprint_library()
    for bp in bp_lib:
        if bp.id.startswith("blueprint."):
            list_static_anomalies.append(bp.id.split(".")[1])

    for anomaly_name in list_static_anomalies:
        if "stop" in anomaly_name:
            continue
        anomaly_name_bp: carla.ActorBlueprint = bp_lib.filter(f"blueprint.{anomaly_name}")[0]
        ano_actor = world.try_spawn_actor(anomaly_name_bp,carla.Transform(carla.Location(0,0,100), carla.Rotation(0,0,0)))
        if ano_actor is None:
            print(f"Could not spawn anomaly {anomaly_name}")
            continue
        width = ano_actor.bounding_box.extent.y * 2
        height = ano_actor.bounding_box.extent.z * 2
        length = ano_actor.bounding_box.extent.x * 2
        area_bb = 2*(length*width + length*height + width*height)
        print(anomaly_name, area_bb, ano_actor.bounding_box.extent)
        if area_bb < 0.25:
            print(anomaly_name, "TINY")
        if area_bb > 0.25 and area_bb < 1.3:
            print(anomaly_name, "SMALL")
        if area_bb > 1.3 and area_bb < 7.0:
            print(anomaly_name, "MEDIUM")
        if area_bb > 7.0:
            print(anomaly_name, "LARGE")
        ano_actor.destroy()

if __name__ == '__main__':
    find_size_anomaly()