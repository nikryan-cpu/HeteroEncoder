import os
import re
import sys
import glob
import subprocess
import pandas as pd
from pymol import cmd
from chemplus import mol_df
from io import StringIO

binana_exe_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep + "exe"

def run_binana(lig_pdbqt, protein_pdbqt, output_dir):
    if not os.path.exists(lig_pdbqt):
        raise Exception("Wrong path to the ligand pdbqt file")
    
    if not os.path.exists(protein_pdbqt):
        raise Exception("Wrong path to the protein pdbqt file")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    path_to_binana = binana_exe_dir + os.sep + "binana_v1dot3.py"
    
    cmd_args = [sys.executable, path_to_binana, '-receptor', protein_pdbqt, '-ligand', lig_pdbqt, '-output_dir', output_dir]
    result = subprocess.run(cmd_args, encoding="utf-8", capture_output=True)
    
    if result.returncode:
        raise Exception(result.stderr)
    else:
        return result.stdout

def parse_binana_log(binana_log_file, out_csv_path):
    if not os.path.exists(binana_log_file):
        raise Exception("Wrong path to binana log file")
    
    if os.path.isdir(os.path.dirname(out_csv_path)) and not os.path.exists(os.path.dirname(out_csv_path)):
        raise Exception("Out CSV directory name does not exist")
    
    path_to_parser = binana_exe_dir + os.sep + "binana_log_parser.py"
    
    cmd_args = [sys.executable, path_to_parser, '--input_file', binana_log_file, '--out_csv', out_csv_path]
    result = subprocess.run(cmd_args, encoding="utf-8", capture_output=True)
    
    if result.returncode:
        raise Exception(result.stderr)
    else:
        return result.stdout

def binana_analyze(lig_pdbqt, protein_pdbqt, binana_out_dir, out_csv_path=None):
    run_binana(lig_pdbqt, protein_pdbqt, binana_out_dir)
    
    binana_log_file = binana_out_dir + os.sep + "log.txt"
    if out_csv_path is None:
        out_csv_path = binana_out_dir + os.sep + "analysis.csv"
    parse_binana_log(binana_log_file, out_csv_path)

def binana_multianalyze(ligs_pdbqt_dir, protein_pdbqt, binana_out_dir, out_csv_path, lig_names_list=None):
    if not os.path.isdir(ligs_pdbqt_dir):
        raise Exception("Wrong path to ligands directory")
    
    if not os.path.isfile(protein_pdbqt):
        raise Exception("Wrong path to the protein pdbqt file")
    
    if not os.path.isdir(binana_out_dir):
        os.makedirs(binana_out_dir)
    
    if lig_names_list is not None and isinstance(lig_names_list, list):
        if not lig_names_list:
            raise Exception("lig_names_list is empty")
        missed_ligs = []
        for lig_name in lig_names_list:
            if not os.path.isfile(f"{ligs_pdbqt_dir}{os.sep}{lig_name}.pdbqt"):
                missed_ligs.append(lig_name)
        
        if len(missed_ligs):
            raise Exception("The following ligands are not presented in the directory: " + ", ".join(missed_ligs))
        
        selected_ligs = [f"{ligs_pdbqt_dir}{os.sep}{lig_name}.pdbqt" for lig_name in lig_names_list]
    else:
        selected_ligs = glob.glob(f"{ligs_pdbqt_dir}{os.sep}*.pdbqt")
    
    binana_df_list = []
    for lig_path in selected_ligs:
        lig_name = os.path.basename(lig_path).split(".")[0]
        
        binana_out_lig_dir = binana_out_dir + os.sep + lig_name
        binana_analyze(lig_path, protein_pdbqt, binana_out_lig_dir)
        
        df_lig_binana = pd.read_csv(binana_out_lig_dir + os.sep + "analysis.csv")
        df_lig_binana.insert(loc=0, column="Name", value=[lig_name])
        #df_lig_binana["Name"] = [lig_name]
        binana_df_list.append(df_lig_binana)
    
    df_results = pd.concat(binana_df_list)
    
    df_results.to_csv(out_csv_path, index=False)

def split_string_on_chains(string):
    if ":" not in string:
        return [("A", string)]
    
    splitted_string = []
    chain_index = string.find(":")
    while True:
        if ":" in string[chain_index+1:]:
            next_chain_index = chain_index+1 + string[chain_index+1:].find(":")
            splitted_string.append((string[chain_index-1], string[chain_index+1:next_chain_index-3]))
            chain_index = next_chain_index
        else:
            splitted_string.append((string[chain_index-1], string[chain_index+1:]))
            break
    
    return splitted_string

def select_interactions(selection_name, interactions_column):
    for state_idx, interactions in enumerate(interactions_column):
        if pd.isna(interactions):
            continue

        for chain_name, residue_idx in re.findall("([A-Z]):[A-Z]+(\d*)(?:,|$|\()", interactions):
            cmd.select(selection_name, f"prot and state {state_idx+1} and c. {chain_name} and i. {residue_idx}", enable=0, merge=1)

def vizualize_interactions(protein_file, ligs_docked_sdf, ligs_pdbqt_dir, binana_analysis_csv, pymol_out_session, weak_contacts_threshold=2, lig_names_list=None):
    """ write pymol_out_session in .pse format """
    if not os.path.isfile(protein_file):
        raise Exception("Wrong path to the protein file")
    
    if not os.path.isfile(ligs_docked_sdf):
        raise Exception("Wrong path to the ligands file")
    
    if not os.path.isdir(ligs_pdbqt_dir):
        raise Exception("Wrong path to the ligands pdbqt directory")
    
    if not os.path.isfile(binana_analysis_csv):
        raise Exception("Wrong path to the binana results csv file")
    
    cmd.reinitialize()
    cmd.bg_color("white")
    
    if lig_names_list is not None: # the check of selected ligs has already been done in binana_multianalyze
        sel_ligs_df = mol_df.df_from_sdf(ligs_docked_sdf, mol_names=lig_names_list)
        string_io = StringIO()
        mol_df.df_to_sdf(sel_ligs_df, string_io)
        cmd.read_sdfstr(string_io.getvalue(), name="ligs")
        string_io.close()
    else:
        cmd.load(ligs_docked_sdf, "ligs")

    lig_names = [cmd.get_title("ligs", state_idx + 1) for state_idx in range(cmd.count_states("ligs"))]

    df_interactions = pd.read_csv(binana_analysis_csv, index_col="Name")

    missed_ligs = [lig_name for lig_name in lig_names if lig_name not in df_interactions.index]
    if len(missed_ligs):
        raise Exception("The following ligands are not presented in the binana csv file: " + ", ".join(missed_ligs))

    missed_ligs = [lig_name for lig_name in lig_names if not os.path.exists(f'{ligs_pdbqt_dir}{os.sep}{lig_name}.pdbqt')]
    if len(missed_ligs):
        raise Exception("The following ligands are not presented in the pdbqt directory: " + ", ".join(missed_ligs))

    df_interactions = df_interactions.loc[lig_names]

    for state_idx, lig_name in enumerate(lig_names):
        cmd.load(protein_file, "prot", discrete=1)
        cmd.set_title("prot", state_idx + 1, lig_name)
        cmd.load(ligs_pdbqt_dir + os.sep + lig_name + ".pdbqt", "ligs_pdbqt", discrete=1, state=state_idx+1)
        cmd.set_title("ligs_pdbqt", state_idx + 1, lig_name)

    cmd.dss()

    #hydrogen bonds
    for state_idx, hydrogen_bonds in enumerate(df_interactions["Hydrogen bonds"]):
        if pd.isna(hydrogen_bonds):
            continue

        for chain_name, text in split_string_on_chains(hydrogen_bonds):
            for ligand_part, protein_part, residue_idx in re.findall(" (.*?)\.\.\.\*\*?(.*?)\[[A-Z]+(\d*)\]", text):
                if "_" in ligand_part:
                    #ligand is donor
                    lig_atom = ligand_part.split("_")[1]
                    protein_atom = protein_part
                else:
                    #protein is donor
                    lig_atom = ligand_part
                    protein_atom = protein_part.split("_")[0]
                cmd.distance("hydrogen_bonds", f"ligs_pdbqt and n. {lig_atom}", f"prot and c. {chain_name} and i. {residue_idx} and n. {protein_atom}", width=5, label=0, reset=0, state=state_idx+1)
                cmd.select("hydrogen_bonds_residues", f"prot and state {state_idx+1} and c. {chain_name} and i. {residue_idx}", enable=0, merge=1)
                #print(state_idx, chain_name, residue_idx)

    #hydrophobic contacts
    for state_idx, hydrophobic_contacts in enumerate(df_interactions["Hydrophobic contacts (C-C)"]):
        if pd.isna(hydrophobic_contacts):
            continue

        for chain_name, text in split_string_on_chains(hydrophobic_contacts):
            for residue_idx, count in re.findall(" [A-Z]+(\d*)\((\d+)", text):
                if int(count) >= weak_contacts_threshold:
                    cmd.select("hydrophobic_contacts_residues", f"prot and state {state_idx+1} and c. {chain_name} and i. {residue_idx}", enable=0, merge=1)
                else:
                    cmd.select("weak_hydrophobic_contacts_residues", f"prot and state {state_idx+1} and c. {chain_name} and i. {residue_idx}", enable=0, merge=1)

    #pi-pi stacking
    if not df_interactions["pi-pi stacking interactions"].isna().all():
        select_interactions("pipi_stacking", df_interactions["pi-pi stacking interactions"])
        cmd.show("lines", "(pipi_stacking)")

    #T-stacking
    if not df_interactions["T-stacking (face-to-edge) interactions"].isna().all():
        select_interactions("t_stacking", df_interactions["T-stacking (face-to-edge) interactions"])
        cmd.show("lines", "(t_stacking)")

    #Cation-pi
    if not df_interactions["Cation-pi interactions"].isna().all():
        select_interactions("cation_pi", df_interactions["Cation-pi interactions"])
        cmd.show("lines", "(cation_pi)")

    #Salt Bridges
    if not df_interactions["Salt Bridges"].isna().all():
        select_interactions("sald_bridges", df_interactions["Salt Bridges"])
        cmd.show("lines", "(sald_bridges)")

    cmd.orient("ligs")
    cmd.color("green", "prot and elem C")
    cmd.color("lightmagenta", "ligs and elem C")
    if not df_interactions["Hydrogen bonds"].isna().all():
        cmd.color("black", "hydrogen_bonds")

    if not df_interactions["Hydrophobic contacts (C-C)"].isna().all():
        cmd.show("sticks", "(hydrophobic_contacts_residues)")
        if weak_contacts_threshold > 0:
            cmd.show("sticks", "(weak_hydrophobic_contacts_residues)")
    if not df_interactions["Hydrogen bonds"].isna().all():
        cmd.show("sticks", "(hydrogen_bonds_residues)")

    cmd.disable("ligs_pdbqt")
    cmd.hide("sticks", "ligs and elem H and nbr. elem C")
    cmd.set("stick_h_scale", 1)
    cmd.set("stick_radius", 0.25, "ligs")
    cmd.set("stick_radius", 0.15, "prot")
    if not df_interactions["Hydrogen bonds"].isna().all():
        cmd.set("stick_radius", 0.20, "(hydrogen_bonds_residues)")
    if weak_contacts_threshold > 0:
        cmd.set("stick_transparency", 0.3, "(weak_hydrophobic_contacts_residues)")

    s = """ one_letter = {'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \\
            'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    \\
            'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    \\
            'GLY':'G', 'PRO':'P', 'CYS':'C'} """

    cmd.do(s)
    if len(cmd.get_chains("prot")) < 2:
        if not df_interactions["Hydrogen bonds"].isna().all():
            cmd.label("(hydrogen_bonds_residues) and n. CA", "one_letter[resn] + resi")
        cmd.label("(hydrophobic_contacts_residues) and n. CA", "one_letter[resn] + resi")
    else:
        if not df_interactions["Hydrogen bonds"].isna().all():
            cmd.label("(hydrogen_bonds_residues) and n. CA", "chain + ':' + one_letter[resn] + resi")
        cmd.label("(hydrophobic_contacts_residues) and n. CA", "chain + ':' + one_letter[resn] + resi")
    #cmd.set("label_color", "black")
    cmd.set("label_size", 36)
    cmd.set("label_position", [0, 2, 5])
    #cmd.set("float_labels", "off")
    #cmd.set("label_outline_color", "deepblue")
    cmd.set("label_font_id", 10)
    cmd.set("cartoon_transparency", 0.8, "prot")

    cmd.draw()

    cmd.save(pymol_out_session)
    cmd.reinitialize()


def binana_vizualize(protein_pdbqt, ligs_pdbqt_dir, ligs_docked_sdf, binana_out_dir, binana_analysis_csv, pymol_out_session, 
                     lig_names_list=None, weak_contacts_threshold=2):
    if lig_names_list is not None and not isinstance(lig_names_list, list):
        raise Exception("Please provide the parameter 'lig_names_list' in the format of a Python List")
    binana_multianalyze(ligs_pdbqt_dir, protein_pdbqt, binana_out_dir, binana_analysis_csv, lig_names_list=lig_names_list)
    vizualize_interactions(protein_pdbqt, ligs_docked_sdf, ligs_pdbqt_dir, binana_analysis_csv, pymol_out_session, weak_contacts_threshold, lig_names_list=lig_names_list)
