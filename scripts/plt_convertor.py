from plyfile import PlyData, PlyElement
import numpy as np
import os
from pathlib import Path
from argparse import ArgumentParser

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

parser = ArgumentParser("point converter")
parser.add_argument("--path", required=True, type=str)
args = parser.parse_args()

plydata = PlyData.read(args.path)
xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),  axis=1)
opacities = np.asarray(plydata.elements[0]["opacity"])
features_dc = np.zeros((xyz.shape[0], 3))
features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

mask = sigmoid(opacities)>0.2
xyz = xyz[mask]
features_dc = features_dc[mask]
rgb = np.clip(features_dc * 0.28209479177387814 + 0.5, 0.0, 1.0) * 255

dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

normals = np.zeros_like(xyz)
elements = np.empty(xyz.shape[0], dtype=dtype)
attributes = np.concatenate((xyz, normals, rgb), axis=1)
elements[:] = list(map(tuple, attributes))
vertex_element = PlyElement.describe(elements, 'vertex')

plydata_new = PlyData([vertex_element])
path_new = args.path.split('.')[0] + '_new.' + args.path.split('.')[1]
plydata_new.write(path_new)