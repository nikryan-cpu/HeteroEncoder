from rdkit import Chem
from rdkit.Chem import AllChem
from chemplus import mdl
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

def get_embed_configs(seed):
    configs_list = []

    embed_params = AllChem.ETKDGv3()
    if seed is not None:
        embed_params.randomSeed = seed
    embed_params.useSmallRingTorsions = True
    configs_list.append((embed_params, "ETKDGv3 with Small Ring Torsions"))

    embed_params = AllChem.ETKDGv3()
    if seed is not None:
        embed_params.randomSeed = seed
    embed_params.useSmallRingTorsions = True
    embed_params.useRandomCoords = True
    configs_list.append((embed_params, "ETKDGv3 with Small Ring Torsions and Random Coords"))

    embed_params = AllChem.ETKDG()
    if seed is not None:
        embed_params.randomSeed = seed
    configs_list.append((embed_params, "ETKDG"))

    embed_params = AllChem.ETKDG()
    if seed is not None:
        embed_params.randomSeed = seed
    embed_params.useRandomCoords = True
    configs_list.append((embed_params, "ETKDG with Random Coords"))

    return configs_list

def generate_3d(mol, seed=None):
    mol_h = Chem.AddHs(mol)
    log = ""
    if seed is not None:
        log += f"Using seed: {seed}\n"

    configs_list = get_embed_configs(seed)
    for embed_params, config_name in configs_list:
        mol_h_copy = Chem.Mol(mol_h)
        try:
            conf_id = AllChem.EmbedMolecule(mol_h_copy, embed_params)
            if conf_id == -1:
                log += f"Failed with {config_name} 3D generation\n"
                continue
            else:    
                log += f"Successfully generated 3D with {config_name}\n"
        except Exception as e:
            log += f"Failed with {config_name} 3D generation:\n{e}\n"
            continue
        Chem.SanitizeMol(mol_h_copy)
        Chem.rdmolops.AssignStereochemistryFrom3D(mol_h_copy)

        if not mdl.is_iso_mol(mol_h_copy, mol_h):
            log += "Stereochemistry changed in new 3D geometry\n"
            continue

        opti_result = AllChem.MMFFOptimizeMolecule(mol_h_copy, maxIters=100000)
        if opti_result == 1:
            log += "The number of MMFF optimization iterations reached the maximum of 100,000\n"
            continue
        elif opti_result == 0:
            log += "Successfully optimized 3D with MMFF\n"
        elif opti_result == -1:
            log += "The MMFF forcefield could not be set up. Trying UFF.\n"
            uff_opti_result = AllChem.UFFOptimizeMolecule(mol_h_copy, maxIters=100000)
            if uff_opti_result == 1:
                log += "The number of UFF optimization iterations reached the maximum of 100,000\n"
                continue
            log += "Successfully optimized 3D with UFF\n"
        Chem.SanitizeMol(mol_h_copy)
        Chem.rdmolops.AssignStereochemistryFrom3D(mol_h_copy)

        if not mdl.is_iso_mol(mol_h_copy, mol_h):
            log += "Stereochemistry changed after optimization\n"
            continue
        return (mol_h_copy, log[:-1])
    return (None, log[:-1])