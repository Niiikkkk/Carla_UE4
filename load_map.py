import os
import glob
import cv2
import open3d as o3d
import numpy as np

from utils import save_radar


def gen_video(path):
    run = path.split('/')[1]
    sens = path.split('/')[2]
    images = sorted([img for img in os.listdir(path)])
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    os.makedirs("output_video/"+run, exist_ok=True)
    video = cv2.VideoWriter("output_video/"+run+"/"+sens+"video.avi", fourcc, 20, (width, height))

    for image in images:
        img = cv2.imread(os.path.join(path, image))
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
    return

def gen_point_video(path):
    run = path.split('/')[1]
    sens = path.split('/')[2]
    plys = sorted([ply for ply in os.listdir(path) if ply.endswith('.ply')])

    pcd = o3d.io.read_point_cloud(os.path.join(path, plys[0]))
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    # Offscreen renderer setup
    width, height = 800, 600
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0

    renderer.scene.set_background([0, 0, 0, 1])  # black background
    renderer.scene.add_geometry("pcd", pcd, mat)

    extent = np.linalg.norm(bbox.get_extent())
    extent = np.max(extent)

    # apply left offset and look-at config
    offset = np.array([0, 0, 0])  # move left
    distance = extent * 0.1  # distance proportional to object size
    renderer.scene.camera.look_at(center + offset,
                                  center + [0, 0, distance] + offset,
                                  [0, 0, 1])



    # Prepare video writer
    os.makedirs("output_video/"+run, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter("output_video/"+run+"/"+sens+"_pointcloud.avi", fourcc, 20, (width, height))

    for fname in plys:
        pcd = o3d.io.read_point_cloud(os.path.join(path, fname))
        renderer.scene.remove_geometry("pcd")
        renderer.scene.add_geometry("pcd", pcd, mat)

        # Render RGB image
        img_o3d = renderer.render_to_image()
        frame = np.asarray(img_o3d)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)

    video.release()
    return


if __name__ == '__main__':
    path = "output/209/"
    gen_video(path+"rgb")
    #gen_video(path+"depth/logaritmic")
    gen_video(path+"semantic/converted")
    #gen_video(path+"instance")
    gen_point_video(path+"semantic_lidar")
    #gen_point_video(path+"lidar")