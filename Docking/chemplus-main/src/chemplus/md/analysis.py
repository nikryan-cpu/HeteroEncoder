import math
from io import StringIO
import pandas as pd
from chemplus.md import energy
from chemplus.md import rmsd

def get_deltaG_df(names_csv, enthalpy_dir, entropy_dir, frames_per_ns, energy_start_end_interval, 
                  energy_sub_start_end_interval, energy_mean_threshold, controls_num=0, sort_by_energy=True):
    df_names = pd.read_csv(names_csv, dtype={"Name" : str})
    
    df = energy.get_full_energy_df(enthalpy_dir, entropy_dir, frames_per_ns, energy_start_end_interval, 
                                   energy_sub_start_end_interval, names_list=df_names["Name"].values)
    df = df.rename(columns={"Full energy" : "∆G"})
    df = df_names.merge(df, how="left", on="Name")
    
    deltaG_mean_std = df["∆G"].apply(lambda x: energy.get_threshold_mean_std(x, energy_mean_threshold) if type(x) == list else math.nan).apply(pd.Series)
    df["<∆G>"], df["∆Gstd"] = deltaG_mean_std[0], deltaG_mean_std[1]
    
    if not sort_by_energy:
        return df
    
    if controls_num:
        df = pd.concat([df[:controls_num].sort_values(by="<∆G>"), df[controls_num:].sort_values(by="<∆G>")])
    else:
        df = df.sort_values(by="<∆G>")
    return df

def energy_analyze(names_csv, name_column, enthalpy_dir, entropy_dir, frames_per_ns, energy_start_end_interval, 
                   energy_sub_start_end_interval, energy_mean_threshold, energy_table_xlsx, energy_image_file, 
                   language, bar_image_file=None, controls_num=0, names_order_csv=None, sort_by_energy=True):
    df = get_deltaG_df(names_csv, enthalpy_dir, entropy_dir, frames_per_ns, energy_start_end_interval, 
                       energy_sub_start_end_interval, energy_mean_threshold, controls_num, sort_by_energy)
    df = df.reset_index(drop=True)
    
    if names_order_csv:
        df[["Name", name_column]].to_csv(names_order_csv, index=False)
    
    deltaH_mean_std = df["∆G"].apply(lambda x: energy.get_threshold_mean_std(x, energy_mean_threshold, 2) if type(x) == list else math.nan).apply(pd.Series)
    df["<∆H>"], df["∆Hstd"] = deltaH_mean_std[0], deltaH_mean_std[1]
    
    TdeltaS_mean_std = df["∆G"].apply(lambda x: energy.get_threshold_mean_std(x, energy_mean_threshold, 3) if type(x) == list else math.nan).apply(pd.Series)
    df["<T∆S>"], df["T∆Sstd"] = TdeltaS_mean_std[0], TdeltaS_mean_std[1]
    
    df[[name_column, "<∆H>", "∆Hstd", "<T∆S>", "T∆Sstd", "<∆G>", "∆Gstd"]].to_excel(energy_table_xlsx, float_format="%.1f", index=False)
    
    energy.get_energy_plots(df, name_column=name_column, energy_column="∆G", controls_num=controls_num, language=language, 
                            mean_threshold_ns=energy_mean_threshold, mean_line=False, mean_window_ns=10, show_without_smoothing=True, save_file=energy_image_file)
    
    if bar_image_file:
        energy.get_energy_bar(df, name_column=name_column, energy_column="∆G", controls_num=controls_num, 
                              mean_threshold_ns=energy_mean_threshold, language=language, save_file=bar_image_file)
    
    return df
        
def rmsd_analyze(names_csv, name_column, rmsd_dir, rms2d_dir, frames_per_ns, rmsd_start_end_interval, 
                 rms2d_start_end_interval, rmsd_image_file, rms2d_image_file, language="EN", controls_num=0, rmsd_max=None):
    df_names = pd.read_csv(names_csv, dtype={"Name" : str})
    
    df = rmsd.get_rmsd_df(rmsd_dir, frames_per_ns=frames_per_ns, start_end_interval=rmsd_start_end_interval, 
                          names_list=df_names["Name"].values)
    df = df_names.merge(df, how="left", on="Name")
    rmsd.get_rmsd_plots(df, name_column, "RMSD", controls_num=controls_num, language=language, rmsd_max=rmsd_max, 
                        save_file=rmsd_image_file)
    
    df_rms2d = rmsd.get_rmsd_df(rms2d_dir, frames_per_ns=frames_per_ns, start_end_interval=rms2d_start_end_interval, rms2d=True, 
                                names_list=df_names["Name"].values)
    df_rms2d = df_names.merge(df_rms2d, how="left", on="Name")
    rmsd.get_rms2d_plots(df_rms2d, name_column, "RMS2D", controls_num=controls_num, language=language, rmsd_max=rmsd_max, 
                         save_file=rms2d_image_file)
    

def full_analysis(names_csv, name_column, enthalpy_dir, entropy_dir, frames_per_ns, energy_start_end_interval, 
                  energy_sub_start_end_interval, energy_mean_threshold, energy_table_xlsx, energy_image_file, 
                  rmsd_dir, rms2d_dir, rmsd_start_end_interval, rms2d_start_end_interval, rmsd_image_file, 
                  rms2d_image_file, rmsd_max=None, bar_image_file=None, apo_names_csv=None, control_names_csv=None, 
                  names_order_csv=None, language="EN", sort_by_energy=True, reverse_rmsd=False):
    
    df_names = pd.read_csv(names_csv, dtype={"Name" : str})
    if control_names_csv:
        df_control_names = pd.read_csv(control_names_csv, dtype={"Name" : str})
        controls_num = df_control_names.shape[0]
        df_names = pd.concat([df_control_names, df_names])
    else:
        controls_num = 0
    
    energy_analyze(StringIO(df_names.to_csv()), name_column, enthalpy_dir, entropy_dir, frames_per_ns, 
                   energy_start_end_interval, energy_sub_start_end_interval, energy_mean_threshold, energy_table_xlsx, 
                   energy_image_file, language, bar_image_file=bar_image_file, controls_num=controls_num, 
                   names_order_csv=names_order_csv, sort_by_energy=sort_by_energy)
    
    if names_order_csv:
        df_names = pd.read_csv(names_order_csv, dtype={"Name" : str})
    
    if apo_names_csv:
        df_apo_names = pd.read_csv(apo_names_csv, dtype={"Name" : str})
        controls_num += df_apo_names.shape[0]
        df_names = pd.concat([df_apo_names, df_names])
    
    if reverse_rmsd:
        df_names = pd.concat([df_names[controls_num:], df_names[:controls_num]])
        controls_num = df_names.shape[0] - controls_num
    
    rmsd_analyze(StringIO(df_names.to_csv()), name_column, rmsd_dir, rms2d_dir, frames_per_ns, rmsd_start_end_interval, 
                 rms2d_start_end_interval, rmsd_image_file, rms2d_image_file, language=language, controls_num=controls_num, rmsd_max=rmsd_max)
    