import sys

import argparse

import os.path as osp
from glob import glob

import numpy as np
from loguru import logger

import pickle
import numpy as np

def convert_pkl_to_npz(pkl_filepath):
    """
    Converts a .pkl file containing a dictionary of NumPy arrays
    to a .npz file.

    Args:
        pkl_filepath (str): Path to the input .pkl file.
    """
    try:
        # Load data from the PKL file
        with open(pkl_filepath, 'rb') as f:
            pkl_data = pickle.load(f, encoding='latin-1')

        pkl_data['J_regressor'] = pkl_data['J_regressor'].toarray()

        npz_filepath = pkl_filepath.replace('.pkl','.npz')

        np.savez_compressed(npz_filepath, **pkl_data, allow_pickle=True)
        print(f"Successfully converted '{pkl_filepath}' to '{npz_filepath}'.")

    except FileNotFoundError:
        print(f"Error: PKL file not found at '{pkl_filepath}'")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PKL to NPZ converter')

    parser.add_argument('--file', required=True, type=str, help='The path to the file to convert')

    args = parser.parse_args()

    pkl_filepath = args.file

    convert_pkl_to_npz(pkl_filepath)
