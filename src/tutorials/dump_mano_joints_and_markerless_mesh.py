import argparse

import torch
import pickle
import smplx
import json

import chumpy as ch
import numpy as np

from loguru import logger

from psbody.mesh import Mesh

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.rotation_tools import rotate_points_xyz

from moshpp.models.smpl_fast_derivatives import SmplModelLBS
from moshpp.models.smpl_fast_derivatives import load_surface_model

to_cpu = lambda tensor: tensor.detach().cpu().numpy()

# Copied over from MoSH repo and modified for convenience
def dump_stagei_mano_mesh(body, output_mesh_obj_fname):
    verts = c2c(body.v[0])
    faces = c2c(body.f)
    
    body_mesh = Mesh(verts, faces, vc=[.7, .7, .7])
    
    body_mesh.write_obj(output_mesh_obj_fname)

    logger.info(f'created {output_mesh_obj_fname}')

def dump_stagei_mano_joints(body, surface_model_fname, surface_model_type, output_joints_json_fname):
    sm_temp = load_surface_model(surface_model_fname=surface_model_fname,
                            surface_model_type=surface_model_type,
                            pose_hand_prior_fname = None
                            )
    
    verts = c2c(body.v[0])

    rest_hand_global_orient = np.zeros((1, sm_temp.trans.size))
    rest_hand_transl = np.zeros((1, sm_temp.trans.size))
    rest_hand_pose = np.zeros((1, sm_temp.pose.size))
    rest_hand_fullpose = np.zeros((1, sm_temp.fullpose.size))

    rest_params = {
        'global_orient': rest_hand_global_orient,
        'transl': rest_hand_transl,
        'pose': rest_hand_pose,
        'fullpose': rest_hand_fullpose
    }
    
    hand_m = smplx.create(model_path=surface_model_fname,
                model_type=surface_model_type,
                flat_hand_mean=True,
                v_template=verts,
                batch_size=1)

    hand_rest_params = {k: torch.from_numpy(v).type(torch.float32) for k, v in rest_params.items()}
    hand_rest_output = hand_m(**hand_rest_params)
    joints_hand_rest_positions = to_cpu(hand_rest_output.joints)
    joints_hand_rest_positions = joints_hand_rest_positions[0]
    
    joints_hierarchy = np.array([255, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14], dtype=np.uint8)
    num_joints = len(joints_hierarchy)

    hand_relative_rest_configuration = np.zeros(joints_hand_rest_positions.shape, dtype=joints_hand_rest_positions.dtype)
    for ji in range(num_joints):
        parent_index = joints_hierarchy[ji]

        if parent_index == 255:
            hand_relative_rest_configuration[ji] = joints_hand_rest_positions[ji]
        else:
            hand_relative_rest_configuration[ji] = joints_hand_rest_positions[ji] - joints_hand_rest_positions[parent_index]

    joint_json = {
        "joint_positions": hand_relative_rest_configuration.flatten().tolist()
    }
    
    with open(output_joints_json_fname, "w+") as fout:
        json.dump(joint_json, fout, indent=4)
    
    logger.info(f'created {output_joints_json_fname}')

def load_body_model(mosh_stagei_pkl_fname):
    assert mosh_stagei_pkl_fname.endswith('.pkl'), ValueError(f'mosh_stagei_pkl_fname should be a valid pkl file: {mosh_stagei_pkl_fname}')
    mosh_stagei = pickle.load(open(mosh_stagei_pkl_fname, 'rb'))

    surface_model_fname = mosh_stagei['stagei_debug_details']['cfg']['surface_model']['fname']
    surface_model_type = mosh_stagei['stagei_debug_details']['cfg']['surface_model']['type']
    
    surface_model_fname_npz = surface_model_fname.replace('.pkl', '.npz')

    body_params = {}
    if 'betas' in mosh_stagei:
        num_betas = mosh_stagei['stagei_debug_details']['cfg']['surface_model']['num_betas']
        body_params['betas'] =  torch.Tensor(mosh_stagei['betas'][:num_betas][None])
        body_params['num_betas'] =  num_betas
    
    bm = BodyModel(surface_model_fname_npz, num_betas=body_params.get('num_betas', 10), model_type=surface_model_type)
    body = bm(**body_params)
    
    return surface_model_fname, surface_model_type, body

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bathing-Dataset-MANO-MOSH')

    parser.add_argument('--path', required=True, type=str, help='The path to the MANO stage_ii PKL to extract')

    args = parser.parse_args()

    path = args.path
    
    output_mesh_obj_fname = path.replace('.pkl', '_markerless.obj')
    output_joints_json_fname = path.replace('.pkl', '_joints.hj')
    
    surface_model_fname, surface_model_type, body = load_body_model(path)
    
    dump_stagei_mano_mesh(body, output_mesh_obj_fname)
    dump_stagei_mano_joints(body, surface_model_fname, surface_model_type, output_joints_json_fname)
