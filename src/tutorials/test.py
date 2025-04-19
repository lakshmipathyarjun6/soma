import os.path as osp
from glob import glob

import numpy as np
from loguru import logger

from soma.amass.mosh_manual import mosh_manual

soma_work_base_dir = '/home/arjun/Desktop/SOMA_Test'
support_base_dir = osp.join(soma_work_base_dir, 'support_files')

mocap_base_dir = osp.join(support_base_dir, 'evaluation_mocaps/original')

work_base_dir = osp.join(soma_work_base_dir, 'running_just_mosh')

render_base_dir = osp.join(work_base_dir, 'render_results')

target_ds_names = ['SOMA_manual_labeled',]

for ds_name in target_ds_names:
    mocap_fnames = glob(osp.join(mocap_base_dir, ds_name,  '*/*.c3d'))

    logger.info(f'#mocaps found for {ds_name}: {len(mocap_fnames)}')

    mosh_manual(
        mocap_fnames,
        mosh_cfg={
            'moshpp.verbosity': 1, # set to 2 to visulaize the process in meshviewer
            'dirs.work_base_dir': osp.join(work_base_dir, 'mosh_results'),
            'dirs.support_base_dir': support_base_dir,
        },
        render_cfg={
            'dirs.work_base_dir': osp.join(work_base_dir, 'render_results'),
            'dirs.mesh_out_dir': osp.join(render_base_dir, 'meshes'),
            'dirs.png_out_dir': osp.join(render_base_dir, 'pngs'),
            'render.render_engine': 'eevee',  # eevee / cycles,
            # 'render.render_engine': 'cycles',  # eevee / cycles,
            'render.show_markers': True,
            'render.save_final_blend_file': True,
            'dirs.support_base_dir': support_base_dir,

        },
        parallel_cfg={
            'pool_size': 1,
            'max_num_jobs': 1,
            'randomly_run_jobs': True,
        },
        run_tasks=[
            # 'mosh',
            'render'
        ],
        # fast_dev_run=True,
    )
