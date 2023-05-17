#!/usr/bin/env python3

try:
    import bpy
except ModuleNotFoundError:
    print(
        "This program needs to be executed through blender like this:\n"
        "```\nblender --background --python export_blender.py -- --output_blend /tmp/cameras.blend --help\n```"
    )
    exit()

import argparse
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy.types import Collection, Scene
from bpy_extras.image_utils import load_image
from mathutils import Matrix, Vector

sys.path.append(os.fspath(Path(__file__).resolve().parent.parent / "dataset"))
from camera_data import CameraData, read_calibration_csv


def add_to_collection(collection: Collection, blender_object):
    if blender_object.name not in collection.objects:
        collection.objects.link(blender_object)


def add_camera_renderview(scene: Scene, camera: CameraData):
    if f"renderview_{camera.name}" not in scene.render.views:
        renderview = scene.render.views.new(f"renderview_{camera.name}")
        renderview.camera_suffix = f"_{camera.name}"


def create_pinhole_camera(
    image_width: float,
    image_height: float,
    fx: float,
    cx: float,
    cy: float,
    collection: Collection,
    name_data: str = "camera_data",
    name_object: str = "camera_object",
    exist_ok: bool = True,
):
    if not exist_ok and (name_data in bpy.data.cameras or name_object in bpy.data.objects):
        raise RuntimeError(f"Camera already exists name_data={name_data}, name_data={name_data}")

    camera_data = bpy.data.cameras.new(name_data)
    camera_data.sensor_fit = "HORIZONTAL"

    # The unit for shift is relative to image-width or image-height,
    # depending on having sensor-fit "HORIZONTAL" or "VERTICAL"
    camera_data.shift_x = -(cx - 0.5)
    camera_data.shift_y = (cy - 0.5) * image_height / image_width
    camera_data.type = "PERSP"
    camera_data.lens_unit = "MILLIMETERS"
    camera_data.sensor_width = 36
    camera_data.lens = fx * camera_data.sensor_width

    camera = bpy.data.objects.new(name_object, camera_data)
    collection.objects.link(camera)
    return camera


def setup_scene(
    scene: Scene,
    resolution_x: int,
    resolution_y: int,
    add_depth_node: bool = True,
    add_normal_node: bool = True,
    add_mask_node: bool = True,
):
    scene.render.filepath = "//rgb/"
    scene.frame_start = 1
    scene.frame_end = 1
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.use_multiview = True
    scene.render.views_format = "MULTIVIEW"
    scene.render.views["right"].use = False
    scene.render.views["left"].use = False
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.pixel_aspect_x = 1
    scene.render.pixel_aspect_y = 1
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = True

    scene.use_nodes = add_depth_node or add_normal_node
    render_layer = scene.node_tree.nodes.get("Render Layers")
    view_layer = scene.view_layers[0]

    if add_depth_node:
        view_layer.use_pass_z = True
        depth_output = render_layer.outputs["Depth"]
        depth_output_node = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        depth_output_node.name = "Depth Output"
        depth_output_node.format.color_mode = "RGB"
        depth_output_node.format.file_format = "OPEN_EXR"
        depth_output_node.base_path = "//depth"
        depth_output_node.location = Vector((300, 100))
        depth_output_node.mute = True
        scene.node_tree.links.new(depth_output, depth_output_node.inputs["Image"])

    if add_normal_node:
        view_layer.use_pass_normal = True
        normal_output = render_layer.outputs["Normal"]
        normal_output_node = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        normal_output_node.name = "Normal Output"
        normal_output_node.format.color_mode = "RGB"
        normal_output_node.format.file_format = "OPEN_EXR"
        normal_output_node.base_path = "//normal"
        normal_output_node.location = Vector((300, -50))
        normal_output_node.mute = True
        scene.node_tree.links.new(normal_output, normal_output_node.inputs["Image"])

    if add_mask_node:
        view_layer.use_pass_object_index = True
        mask_output = render_layer.outputs["Alpha"]
        mask_output_node = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        mask_output_node.name = "Mask Output"
        mask_output_node.format.color_mode = "BW"
        mask_output_node.format.file_format = "PNG"
        mask_output_node.base_path = "//mask"
        mask_output_node.location = Vector((300, 250))
        mask_output_node.mute = True
        scene.node_tree.links.new(mask_output, mask_output_node.inputs["Image"])


def main():
    parser = argparse.ArgumentParser(description="Program to export cameras to blender.")
    parser.add_argument("--csv", type=Path, help="Path to input csv file.", required=True)
    parser.add_argument("--blend", type=Path, help="Path to output blender scene.", required=True)
    parser.add_argument("--images", type=Path, help="If provided, camera background-image property will be set.")
    parser.add_argument("--image_name", type=str, default="{camera_name}_000000.jpg", help="File name for frames.")
    parser.add_argument(
        "--no_root",
        action="store_true",
        help="Load scene without placing root transformation node which scales scene to meters and rotates to Z up. Consider running `bpy.context.space_data.overlay.show_relationship_lines = False` to hide relationship lines.",
    )
    parser_mesh_group = parser.add_mutually_exclusive_group()
    parser_mesh_group.add_argument("--obj", type=Path, help="Optional mesh to load (wavefront).")
    parser_mesh_group.add_argument("--abc", type=Path, help="Optional mesh to load (alembic).")
    parser_mesh_group.add_argument("--scale", type=float, help="Scale applied to the scene.", default=1.0)
    parser.add_argument("--abc_object_path", default="/object", help="Object path of mesh within abc file.")
    # parser.add_argument("--object_rotation_x", default=, help="Object path of mesh within abc file.")
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    bpy.ops.wm.read_homefile(use_empty=True)

    # save immediately to support paths relative to blend file
    bpy.ops.wm.save_as_mainfile(filepath=os.fspath(args.blend))

    cameras = read_calibration_csv(args.csv)
    short_edge = min(cameras[0].width, cameras[0].height)
    long_edge = max(cameras[0].width, cameras[0].height)

    # setup 2 scenes (one for landscape, one for portrait cameras)
    assert len(bpy.data.scenes) == 1
    default_scene = bpy.data.scenes[0]
    scene_landscape = bpy.data.scenes.new("landscape")
    scene_portrait = bpy.data.scenes.new("portrait")
    setup_scene(scene_landscape, long_edge, short_edge)
    setup_scene(scene_portrait, short_edge, long_edge)
    bpy.data.scenes.remove(default_scene)

    # setup 2 collections (one for landscape, one for portrait cameras)
    collection_landscape = bpy.data.collections.new("cameras_landscape")
    collection_portrait = bpy.data.collections.new("cameras_portrait")
    scene_landscape.collection.children.link(collection_landscape)
    scene_portrait.collection.children.link(collection_portrait)

    # add root transformation, so that scene-up is Z and scale in meters
    if not args.no_root:
        root = bpy.data.objects.new("root", None)
        root.empty_display_size = 2
        root.empty_display_type = "PLAIN_AXES"
        collection_landscape.objects.link(root)
        collection_portrait.objects.link(root)
        root.scale = Vector((args.scale, args.scale, args.scale))
        root.rotation_euler = Vector((0.5 * math.pi, 0, 0))
        root.empty_display_size = 1 / args.scale  # Will result in 1m due to scale

    for camera in cameras:
        scene = scene_landscape if camera.height < camera.width else scene_portrait
        collection_cameras = scene.collection
        add_camera_renderview(scene, camera)

        # Square pixels are assumed. If this exception is thrown, you likely use downscaled images.
        assert np.isclose(camera.fx_pixel, camera.fy_pixel)

        blender_camera = create_pinhole_camera(
            camera.width,
            camera.height,
            fx=camera.focal_length[0],
            cx=camera.principal_point[0],
            cy=camera.principal_point[1],
            collection=collection_cameras,
            name_data=f"camd_{camera.name}",
            name_object=f"cam_{camera.name}",
            exist_ok=False,
        )
        scene.camera = blender_camera

        if not args.no_root:
            blender_camera.parent = root
        blender_camera.data.display_size = 0.1 / args.scale
        blender_camera.location = Vector(camera.translation)
        angle = np.linalg.norm(camera.rotation_axisangle)
        rotation = Matrix.Rotation(angle, 4, camera.rotation_axisangle / angle) @ Matrix.Rotation(math.pi, 4, "X")
        blender_camera.rotation_mode = "QUATERNION"
        blender_camera.rotation_quaternion = rotation.to_quaternion()

        if args.images:
            blender_camera.data.show_background_images = True
            camera_image_name = args.image_name.format(camera_name=camera.name)
            image = load_image(camera_image_name, args.images / camera.name, recursive=False, place_holder=True)
            background_image = blender_camera.data.background_images.new()
            background_image.image = image
            image.filepath_raw = f"//{os.path.relpath(args.images  / camera.name / camera_image_name, Path(bpy.data.filepath).parent)}"

    if args.obj:
        collection_person = bpy.data.collections.new("person")
        bpy.ops.import_scene.obj(filepath=str(args.obj))
        person = bpy.context.selected_objects[0]
        person.name = "person"
        person.rotation_euler = Vector((0, 0, 0))
        if not args.no_root:
            person.parent = root
        add_to_collection(scene_landscape.collection, person)
        add_to_collection(scene_portrait.collection, person)

    if args.abc:
        mesh = bpy.data.meshes.new("person")
        person = bpy.data.objects.new("person", mesh)
        person.rotation_euler = (-math.pi / 2, 0, 0)
        collection_person = bpy.data.collections.new("person")
        bpy.ops.cachefile.open(filepath=str(args.abc))
        cache_file = bpy.data.cache_files[0]
        sequence_cache = person.modifiers.new("sequence_cache", "MESH_SEQUENCE_CACHE")
        sequence_cache.cache_file = cache_file
        sequence_cache.object_path = (
            args.abc_object_path
        )
        sequence_cache.use_vertex_interpolation = False
        if not args.no_root:
            person.parent = root
        add_to_collection(scene_landscape.collection, person)
        add_to_collection(scene_portrait.collection, person)

    bpy.ops.wm.save_as_mainfile(filepath=os.fspath(args.blend))


if __name__ == "__main__":
    main()
