import glob
import os
import os.path as osp
import shutil

import numpy as np
import json

from loguru import logger

from soma.render.blender_tools import prepare_render_cfg

# Avoid dealing with Blender inconsistencies - just load from file
def load_body_shape_from_obj(body_mesh_fname):
    try:
        vertices = []
        with open(body_mesh_fname) as f:
            for line in f:
                if line[0] == "v":
                    vertex = list(map(float, line[2:].strip().split()))
                    vertices.append(vertex)

        vertices_np = np.array(vertices)
        flattened_verts = vertices_np.flatten().tolist()

        logger.success(f'loaded {body_mesh_fname}')

        return flattened_verts

    except FileNotFoundError:
        print(f"{body_mesh_fname} not found.")
    except:
        print("An error occurred while loading the shape.")

def create_export_sequence_from_mesh_dir(cfg):
    cfg = prepare_render_cfg(**cfg)

    logger.debug(f'input mesh dir: {cfg.dirs.mesh_out_dir}')

    body_mesh_fnames = sorted(glob.glob(os.path.join(cfg.dirs.mesh_out_dir, 'body_mesh', '*.obj')))
    assert len(body_mesh_fnames)

    all_entries = []

    for body_mesh_fname in body_mesh_fnames:
        flattened_verts = load_body_shape_from_obj(body_mesh_fname)

        frame_id_str = body_mesh_fname.split('/')[-1].split('.')[0]
        frame_id = int(frame_id_str)

        entry = {
            'frame': frame_id,
            'vertices': flattened_verts
        }

        all_entries.append(entry)

    out_mp4_fname = cfg.dirs.mp4_out_fname
    out_json_fname = out_mp4_fname.replace('.mp4', '.smplxmosh')
    out_obj_fname = out_mp4_fname.replace('.mp4', '_basemesh.obj')

    with open(out_json_fname, "w+") as fout:
        json.dump(all_entries, fout, indent=4)

    shutil.copyfile(body_mesh_fnames[0], out_obj_fname)
