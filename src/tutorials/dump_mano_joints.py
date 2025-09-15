import argparse
import os

import torch
import pickle

from loguru import logger

from moshpp.mosh_head import MoSh

from psbody.mesh import Mesh

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

# Copied over from MoSH repo and modified for convenience
def dump_stagei_mano_mesh(mosh_stagei_pkl_fname):
    assert mosh_stagei_pkl_fname.endswith('.pkl'), ValueError(f'mosh_stagei_pkl_fname should be a valid pkl file: {mosh_stagei_pkl_fname}')
    mosh_stagei = pickle.load(open(mosh_stagei_pkl_fname, 'rb'))

    output_ply_fname = mosh_stagei_pkl_fname.replace('.pkl', '_markerless.ply')

    surface_model_fname = mosh_stagei['stagei_debug_details']['cfg']['surface_model']['fname']
    if surface_model_fname.endswith('.pkl'):
        surface_model_fname = surface_model_fname.replace('.pkl', '.npz')

    body_parms = {}
    if 'betas' in mosh_stagei:
        num_betas = mosh_stagei['stagei_debug_details']['cfg']['surface_model']['num_betas']
        body_parms['betas'] =  torch.Tensor(mosh_stagei['betas'][:num_betas][None])
        body_parms['num_betas'] =  num_betas
    surface_model_type = mosh_stagei['stagei_debug_details']['cfg']['surface_model']['type']
    
    bm = BodyModel(surface_model_fname, num_betas=body_parms.get('num_betas', 10), model_type=surface_model_type)
    body = bm(**body_parms)

    verts = c2c(body.v[0])
    faces = c2c(body.f)
    
    body_mesh = Mesh(verts, faces, vc=[.7, .7, .7])
    
    body_mesh.write_ply(output_ply_fname)

    logger.info(f'created {output_ply_fname}')

# def dump_mano_joints(bathing_work_base_dir, path, hand):
#     support_base_dir = os.path.join(bathing_work_base_dir, 'support_files')
#     MANO_hand = 'MANO_RIGHT' if hand == 'right' else 'MANO_LEFT'

#     stageii_data = pickle.load(open(path, 'rb'))
    
#     mosh_result = MoSh.load_as_amass_npz(path, include_markers=True)
    
#     surface_model_fname = os.path.join(support_base_dir, 'mano', 'models', '{}.npz'.format(MANO_hand))
#     assert os.path.exists(surface_model_fname), FileExistsError(surface_model_fname)

#     num_betas = len(mosh_result['betas']) if 'betas' in mosh_result else 10
#     num_dmpls = None
#     dmpl_fname = None
#     num_expressions = len(mosh_result['expression']) if 'expression' in mosh_result else None
    
#     sm = BodyModel(bm_fname=surface_model_fname,
#                 num_betas=num_betas,
#                 num_expressions=num_expressions,
#                 num_dmpls=num_dmpls,
#                 dmpl_fname=dmpl_fname)

    
#     fullposes = stageii_data["fullpose"]
    
#     print(stageii_data['n_comps'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bathing-Dataset-MANO-MOSH')

    # parser.add_argument('--data-dir', required=True, type=str, help='The path to the top-level data directory')
    parser.add_argument('--path', required=True, type=str, help='The path to the MANO stage_ii PKL to extract')
    # parser.add_argument('--hand', required=True, type=str, help='The name of the hand to use. Can be "left" or "right", defaults to right')

    args = parser.parse_args()

    # bathing_work_base_dir = args.data_dir
    path = args.path
    # hand = args.hand
    
    dump_stagei_mano_mesh(path)
