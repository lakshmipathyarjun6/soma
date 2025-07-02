import sys

import argparse

import os.path as osp
from glob import glob

import numpy as np
from loguru import logger

from soma.amass.mosh_manual import mosh_manual

def run_mosh(bathing_work_base_dir, captures, session, task, hand):
    support_base_dir = osp.join(bathing_work_base_dir, 'support_files')
    mocap_base_dir = osp.join(support_base_dir, 'evaluation_mocaps', captures)

    work_base_dir = osp.join(bathing_work_base_dir, 'running_just_mosh')
    render_base_dir = osp.join(work_base_dir, 'render_results')

    mocap_fnames = glob(osp.join(mocap_base_dir, session,  '*/*.c3d'))

    logger.info(f'#mocaps found for {session}: {len(mocap_fnames)}')

    MANO_hand = 'MANO_RIGHT' if hand == 'right' else 'MANO_LEFT'

    mosh_manual(
        mocap_fnames,
        mosh_cfg={
            'moshpp.verbosity': 1, # set to 2 to visulaize the process in meshviewer
            'dirs.work_base_dir': osp.join(work_base_dir, 'mosh_results'),
            'dirs.support_base_dir': support_base_dir,
            'surface_model.type': 'mano',
            'surface_model.fname': osp.join(support_base_dir, 'mano', 'male', '{}.pkl'.format(MANO_hand)),
            'moshpp.pose_body_prior_fname': None,
            'moshpp.optimize_fingers': True,
            'moshpp.optimize_betas': True,
            'moshpp.head_marker_corr_fname': None,
            'moshpp.stagei_frame_picker.num_frames': 20
        },
        render_cfg={
            'dirs.work_base_dir': osp.join(work_base_dir, 'render_results'),
            'render.render_engine': 'eevee',  # eevee / cycles,
            'render.show_markers': True,
            'render.save_final_blend_file': False,
            'render.compute_meshes_only': True,
            'surface_model.type': 'mano',
            'surface_model.fname': osp.join(support_base_dir, 'mano', 'male', '{}.npz'.format(MANO_hand)),
            'dirs.support_base_dir': support_base_dir,
        },
        parallel_cfg={
            'pool_size': 1,
            'max_num_jobs': 1,
            'randomly_run_jobs': True,
        },
        run_tasks=[
            task
        ]
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bathing-Dataset-MANO-MOSH')

    parser.add_argument('--data-dir', required=True, type=str, help='The path to the top-level data directory')
    parser.add_argument('--captures', required=True, type=str, help='The name of the captures type to run')
    parser.add_argument('--session', required=True, type=str, help='The name of the session to run')
    parser.add_argument('--task', required=False, default='mosh', type=str, help='Task to run. Can be "mosh" or "render", defaults to mosh')
    parser.add_argument('--hand', required=True, type=str, help='The name of the hand to use. Can be "left" or "right", defaults to right')

    args = parser.parse_args()

    bathing_work_base_dir = args.data_dir
    captures = args.captures
    session = args.session
    task = args.task
    hand = args.hand

    run_mosh(bathing_work_base_dir, captures, session, task, hand)
