import glob
import os
import os.path as osp

from human_body_prior.tools.omni_tools import makepath
from loguru import logger

from soma.render.blender_tools import prepare_render_cfg
from soma.render.blender_tools import setup_scene

def create_export_sequence_from_mesh_dir(cfg):
    cfg = prepare_render_cfg(**cfg)

    makepath(cfg.dirs.png_out_dir)

    setup_scene(cfg)

    logger.debug(f'input mesh dir: {cfg.dirs.mesh_out_dir}')

    body_mesh_fnames = sorted(glob.glob(os.path.join(cfg.dirs.mesh_out_dir, 'body_mesh', '*.obj')))
    assert len(body_mesh_fnames)

    for body_mesh_fname in body_mesh_fnames:
        print(body_mesh_fname)