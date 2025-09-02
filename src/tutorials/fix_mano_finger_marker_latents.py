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
    
    # # LEFT
    # new_offsets = np.array([
    #     [-0.02456166,0.01795637,0.02728882],
    #     [-0.02899615,0.02124384,0.00344353],
    #     [-0.04948129,0.0097033,-0.03088414],
    #     [-0.03971722,0.01753085,-0.01376573],
    #     [ 0.04159599,0.01263921,0.03038124],
    #     [ 0.00423865,0.0156514, 0.02771383],
    #     [ 0.06807348,0.00869515,0.02906883],
    #     [ 0.04499674,0.01512095,-0.00362795],
    #     [ 0.00401199,0.01854031,0.00034721],
    #     [ 0.07347797,0.01194882,-0.00913157],
    #     [-0.00017932,0.00400492,-0.06068394],
    #     [-0.01472504,0.00552219,-0.05170796],
    #     [ 0.01881513,-0.00051826,-0.06847671],
    #     [ 0.03328211,0.01163327,-0.03214325],
    #     [-0.00512095,0.01565419,-0.0236536 ],
    #     [ 0.05775571,0.00908809,-0.03970374],
    #     [-0.04278917,-0.00764899,0.07478778],
    #     [-0.0811247, 0.0047987, 0.04893245],
    #     [-0.01057263,-0.01423214,0.09200074],
    #     [-0.07547414,0.01673407,-0.02285153],
    #     [-0.04684339,0.01359964,0.04201285],
    #     [-0.10856403,0.02201246,0.07438564],
    #     [-0.11689754,-0.00594392,-0.05733694]
    # ])
    
    # # RIGHT
    # new_offsets = np.array([
    #     [0.02456166,0.01795637,0.02728882],
    #     [0.02899615,0.02124384,0.00344353],
    #     [0.04948129,0.0097033,-0.03088414],
    #     [0.03971722,0.01753085,-0.01376573],
    #     [-0.04159599,0.01263921,0.03038124],
    #     [-0.00423865,0.0156514,0.02771383],
    #     [-0.06807348,0.00869515,0.02906883],
    #     [-0.04499674,0.01512095,-0.00362795],
    #     [-0.00401199,0.01854031,0.00034721],
    #     [-0.07347797,0.01194882,-0.00913157],
    #     [0.00017932,0.00400492,-0.06068394],
    #     [0.01472504,0.00552219,-0.05170796],
    #     [-0.01881513,-0.00051826,-0.06847671],
    #     [-0.03328211,0.01163327,-0.03214325],
    #     [0.00512095,0.01565419,-0.0236536],
    #     [-0.05775571,0.00908809,-0.03970374],
    #     [0.04278917,-0.00764899,0.07478778],
    #     [0.0811247,0.0047987,0.04893245],
    #     [0.01057263,-0.01423214,0.09200074],
    #     [0.07547414,0.01673407,-0.02285153],
    #     [0.04684339,0.01359964,0.04201285],
    #     [0.10856403,0.02201246,0.07438564],
    #     [0.11689754,-0.00594392,-0.05733694]
    # ])

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
