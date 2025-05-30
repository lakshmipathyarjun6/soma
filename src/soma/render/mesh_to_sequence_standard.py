import glob
import os
import os.path as osp
import shutil

import numpy as np
import json

import bpy
from human_body_prior.tools.omni_tools import makepath
from loguru import logger

from soma.render.blender_tools import make_blender_silent
from soma.render.blender_tools import prepare_render_cfg
from soma.render.blender_tools import setup_scene

def run_blender_once(cfg, body_mesh_fname):
    make_blender_silent()

    bpy.ops.object.delete({"selected_objects": [obj for colec in bpy.data.collections for obj in colec.all_objects if
                                                obj.name in ['Body', 'Object']]})

    if cfg.render.show_body:
        bpy.ops.import_scene.obj(filepath=body_mesh_fname)

        body = bpy.context.selected_objects[0]

        body.name = 'Body'

        v_world_coords = [(body.matrix_world @ v.co) for v in body.data.vertices]
        plain_verts = np.array([list(vert.to_tuple()) for vert in v_world_coords])
        flattened_verts = plain_verts.flatten().tolist()
    else:
        flattened_verts = []

    bpy.ops.object.delete({"selected_objects": [obj for colec in bpy.data.collections for obj in colec.all_objects if
                                                obj.name in ['Body', 'Object']]})

    logger.success(f'loaded {body_mesh_fname}')

    return flattened_verts


def create_export_sequence_from_mesh_dir(cfg):
    cfg = prepare_render_cfg(**cfg)

    makepath(cfg.dirs.png_out_dir)

    setup_scene(cfg)

    logger.debug(f'input mesh dir: {cfg.dirs.mesh_out_dir}')

    body_mesh_fnames = sorted(glob.glob(os.path.join(cfg.dirs.mesh_out_dir, 'body_mesh', '*.obj')))
    assert len(body_mesh_fnames)

    all_entries = []

    for body_mesh_fname in body_mesh_fnames:
        flattened_verts = run_blender_once(cfg, body_mesh_fname)

        frame_id_str = body_mesh_fname.split('/')[-1].split('.')[0]
        frame_id = int(frame_id_str)

        entry = {
            'rel_frame': frame_id,
            'vertices': flattened_verts
        }

        all_entries.append(entry)

    out_mp4_fname = cfg.dirs.mp4_out_fname
    out_json_fname = out_mp4_fname.replace('.mp4', '.json')
    out_obj_fname = out_mp4_fname.replace('.mp4', '_basemesh.obj')

    with open(out_json_fname, "w+") as fout:
        json.dump(all_entries, fout, indent=4)

    shutil.copyfile(body_mesh_fnames[0], out_obj_fname)
