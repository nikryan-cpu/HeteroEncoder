import os
import argparse
from chemplus.vina import docking

def create_parser():
    """ Function for creating command line arguments parser. """
    parser = argparse.ArgumentParser(description="DOCKING", add_help=True)
    
    parser.add_argument('--work_dir', required=True, help="Working directory")
    parser.add_argument('--receptor', required=True, help="Receptor file in PDBQT format which is in receptors/. (rec.pdbqt)")
    parser.add_argument('--config', required=True, help="Configuration file name which is in configs/. (conf.txt)")
    parser.add_argument('--sdf', required=True, help="SDF file with ligands")
    parser.add_argument('--cpu_count', required=True, type=int, help="CPU count")
    parser.add_argument('--serial', action='store_true', help="Serial run (default: mpi)")
    parser.add_argument('--rewrite', action='store_true', help="Overwrite docking")

    return parser

def main():

    parser = create_parser()
    try:
        args = parser.parse_args()
    except Exception as e:
        print(e)
    
    work_dir = os.path.expanduser(args.work_dir)
    receptor_file = os.path.expanduser(args.receptor)
    config_file = os.path.expanduser(args.config)
    sdf_file = os.path.expanduser(args.sdf)
    docking.sdf_dock_solved_path(work_dir, receptor_file, config_file, sdf_file, args.cpu_count, args.serial, args.rewrite)
    
if __name__ == '__main__':
    main()
    