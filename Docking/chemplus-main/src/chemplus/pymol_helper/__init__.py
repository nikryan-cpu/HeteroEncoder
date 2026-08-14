import os
import glob
import inspect
import argparse
from pymol import cmd

def mol2_to_sdf(mol2_dir, out_sdf):
    if not os.path.exists(mol2_dir):
        raise Exception(".mol2 dir doesn't exists")
    
    sdf_dir = os.path.dirname(out_sdf)
    if sdf_dir != "" and not os.path.exists(sdf_dir):
        os.makedirs(sdf_dir)
    
    for mol2_file in glob.iglob(mol2_dir + os.sep + "*.mol2"):
        cmd.load(mol2_file)

    out_sdf_name = os.path.basename(out_sdf).split(".")[0]
    cmd.join_states(out_sdf_name, selection='(all)', mode=0)
    cmd.save(out_sdf, out_sdf_name, state=0)
    cmd.reinitialize()
    
def sdf_to_mol2(sdf, out_dir):
    if not os.path.exists(sdf):
        raise Exception("SDF doesn't exists")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    cmd.load(sdf)
    sdf_name = os.path.basename(sdf).split(".")[0]
    cmd.split_states(sdf_name)
    cmd.delete(sdf_name)
    object_names = cmd.get_object_list(selection='(all)')
    for name in object_names:
        cmd.save(out_dir + os.sep + name + ".mol2", name)
        cmd.delete(name)
    cmd.reinitialize()