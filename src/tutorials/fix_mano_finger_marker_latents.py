import argparse

import os.path as osp
from glob import glob

import torch
import numpy as np
import pickle
import json

from loguru import logger

from moshpp.mosh_head import MoSh
from moshpp.marker_layout.edit_tools import marker_layout_as_mesh
from moshpp.marker_layout.edit_tools import marker_layout_load
from moshpp.marker_layout.edit_tools import marker_layout_to_c3d
from moshpp.marker_layout.edit_tools import marker_layout_write

# Copied over from MoSH repo and modified for convenience
def dump_stagei_marker_layout(mosh_stagei_pkl_fname,
                                out_marker_layout_fname=None,
                                template_marker_layout_fname: str= None):
    assert mosh_stagei_pkl_fname.endswith('.pkl'), ValueError(f'mosh_stagei_pkl_fname should be a valid pkl file: {mosh_stagei_pkl_fname}')
    mosh_stagei = pickle.load(open(mosh_stagei_pkl_fname, 'rb'))

    marker_meta = MoSh.extract_marker_layout_from_mosh(mosh_stagei,
                                                        template_marker_layout_fname=template_marker_layout_fname)
    if out_marker_layout_fname is None:
        out_marker_layout_fname = mosh_stagei_pkl_fname.replace('.pkl', '.json')

    output_ply_fname = mosh_stagei_pkl_fname.replace('.pkl', '.ply')
    output_c3d_fname = mosh_stagei_pkl_fname.replace('.pkl', '.c3d')

    surface_model_fname = mosh_stagei['stagei_debug_details']['cfg']['surface_model']['fname']
    if surface_model_fname.endswith('.pkl'):
        surface_model_fname = surface_model_fname.replace('.pkl', '.npz')
    marker_layout_write(marker_meta, out_marker_layout_fname)

    body_parms = {}
    if 'betas' in mosh_stagei:
        num_betas = mosh_stagei['stagei_debug_details']['cfg']['surface_model']['num_betas']
        body_parms['betas'] =  torch.Tensor(mosh_stagei['betas'][:num_betas][None])
        body_parms['num_betas'] =  num_betas
    surface_model_type = mosh_stagei['stagei_debug_details']['cfg']['surface_model']['type']
    marker_layout_as_mesh(surface_model_fname,
                            preserve_vertex_order=True,
                            body_parms=body_parms,
                            surface_model_type= surface_model_type)(out_marker_layout_fname,output_ply_fname)
    marker_layout_to_c3d(out_marker_layout_fname,
                            surface_model_fname=surface_model_fname,
                            out_c3d_fname=output_c3d_fname,
                            surface_model_type= surface_model_type)

    logger.info(f'created {out_marker_layout_fname}')
    logger.info(f'created {output_ply_fname}')
    logger.info(f'created {output_c3d_fname}')

def fix_mano_mosh(path):
    stagei_data = pickle.load(open(path, 'rb'))

    markers_latent_vids = stagei_data["markers_latent_vids"]
    markers_latent_offsets = stagei_data["markers_latent"]

    new_offsets = np.zeros(markers_latent_offsets.shape)
    new_offsets[-2:] = markers_latent_offsets[-2:] # keep wrist offsets

    stagei_data["markers_latent"] = new_offsets

    dir_only = path.split('/')
    del dir_only[-1]

    dir_only = '/'.join(dir_only)
    marker_json_path = glob(osp.join(dir_only, '..',  '*.json'))[0]

    with open(marker_json_path, 'r') as file:
        data = json.load(file)

    finger_markers = data['markersets'][0]['indices']

    for key in markers_latent_vids.keys():
        if key in finger_markers:
            stagei_data["markers_latent_vids"][key] = finger_markers[key]

    pickle.dump(stagei_data, open(path, 'wb'))

    dump_stagei_marker_layout(path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bathing-Dataset-MANO-MOSH')

    parser.add_argument('--path', required=True, type=str, help='The path to the MANO stage_i PKL to modify')

    args = parser.parse_args()

    path = args.path

    fix_mano_mosh(path)
