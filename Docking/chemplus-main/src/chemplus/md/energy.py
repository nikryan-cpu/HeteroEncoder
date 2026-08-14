import os
import math
import glob
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from chemplus import md

def get_energy_delta_stringio(dat_file, mode="GB"):
    if not os.path.exists(dat_file):
        raise Exception("'" + dat_file + "' does not exists")
    
    energy_lines = open(dat_file, 'r').readlines()
    
    if mode == "GB":
        mode_line = "GENERALIZED BORN:\n"
    elif mode == "PB":
        mode_line = "POISSON BOLTZMANN:\n"
    elif mode == "NMODE":
        mode_line = "NMODE entropy results\n"
    else:
        raise Exception("Unknown energy mode")
    
    try:
        mode_index = energy_lines.index(mode_line)
    except:
        raise Exception(f"\"{mode_line[:-1]}\" not found in \"{os.path.basename(dat_file).split('.')[0]}\"")
    
    for line_index in range(mode_index + 2, len(energy_lines)):
        if energy_lines[line_index-1:line_index+1] == ["\n", "\n"]:
            mode_end_index = line_index
            break
    
    try:
        delta_index = energy_lines[mode_index:mode_end_index].index("DELTA Energy Terms\n")
    except:
        raise Exception(f"{mode} \"DELTA Energy Terms\" not found in \"{os.path.basename(dat_file).split('.')[0]}\"")
    
    delta_end_index = delta_index + energy_lines[delta_index:].index("\n")
    delta_string = "".join(energy_lines[delta_index+1:delta_end_index]).strip()
    
    return StringIO(delta_string)

def get_energy_df(energy_dir, frames_per_ns, start_end_interval, sub_start_end_interval=None, 
                  energy_type="GB", column_name=None, names_list=None):
    
    #Available energy types: "GB", "PB" and "NMODE"
    if energy_type not in ["GB", "PB", "NMODE"]:
        raise Exception("Unknown energy type")
    
    if not os.path.exists(energy_dir):
        raise Exception("Energy directory does not exists")

    if column_name is not None:
        energy_column = column_name
    else:
        energy_column = energy_type
        
    if energy_type == "NMODE":
        energy_term = "Total"
        if sub_start_end_interval is None:
            raise Exception("Missing entropy intervals")
    else:
        energy_term = "DELTA TOTAL"
    
    energy_time = md.get_time_range(frames_per_ns, start_end_interval, sub_start_end_interval)
    
    pd_list = []
    for energy_file in glob.glob(energy_dir + os.sep + "*.dat"):
        filename = os.path.basename(energy_file).split(".")[0]
        if names_list is not None and filename not in names_list:
            continue
        
        try:
            string_io = get_energy_delta_stringio(energy_file, mode=energy_type)
        except Exception as e:
            print("Skipping file:", e)
            continue
        df_delta = pd.read_csv(string_io)
        energy_values = df_delta[energy_term].values
        if energy_values.shape != energy_time.shape:
            print(f"Skipping file: the number of energy values and frames is not equal in \"{filename}\".")
            continue
        
        energy_points = list(zip(energy_values, energy_time))
        pd_list.append({"Name" : filename, energy_column : energy_points})
    
    return pd.DataFrame(pd_list)

def get_full_energy_points(enthalpy_points, entropy_points):
    if not all(enthalpy_points[i][1] < enthalpy_points[i+1][1] for i in range(len(enthalpy_points) - 1)):
        raise Exception("Enthalpy points are not sorted or contain duplicates")
    if not all(entropy_points[i][1] < entropy_points[i+1][1] for i in range(len(entropy_points) - 1)):
        raise Exception("Entropy points are not sorted or contain duplicates")
    
    full_energy_points = []
    enthalpy_index = 0
    entropy_index = 0
    while enthalpy_index < len(enthalpy_points) and entropy_index < len(entropy_points):
        enthalpy_time_val = enthalpy_points[enthalpy_index][1]
        entropy_time_val = entropy_points[entropy_index][1]
        if enthalpy_time_val < entropy_time_val:
            enthalpy_index += 1
        elif enthalpy_time_val > entropy_time_val:
            entropy_index += 1
        else:
            enthalpy_val = enthalpy_points[enthalpy_index][0]
            entropy_val = entropy_points[entropy_index][0]
            full_energy_points.append((enthalpy_val - entropy_val, entropy_time_val, enthalpy_val, entropy_val))
            entropy_index += 1
            enthalpy_index += 1
    
    return full_energy_points

def get_full_energy(df, enthalpy_column, entropy_column):
    return df[[enthalpy_column, entropy_column]].apply(
        lambda x: get_full_energy_points(x[0], x[1]) if type(x[1]) is list else float("nan"), axis=1)

def get_full_energy_df(enthalpy_dir, entropy_dir, frames_per_ns, start_end_interval, sub_start_end_interval, enthalpy_type="GB", 
                       enthalpy_column=None, entropy_column=None, full_energy_column=None, names_list=None):
    
    df_enthalpy = get_energy_df(enthalpy_dir, frames_per_ns, start_end_interval, 
                                energy_type=enthalpy_type, column_name=enthalpy_column, names_list=names_list)
    df_entropy = get_energy_df(entropy_dir, frames_per_ns, start_end_interval, sub_start_end_interval, 
                               energy_type="NMODE", column_name=entropy_column, names_list=names_list)
    
    df = df_enthalpy.merge(df_entropy, on="Name", how="left")
    
    if enthalpy_column is None:
        enthalpy_column = enthalpy_type
    if entropy_column is None:
        entropy_column = "NMODE"
    if full_energy_column is None:
        full_energy_column = "Full energy"
    
    df[full_energy_column] = get_full_energy(df, enthalpy_column, entropy_column)
    return df

def mean_window(points, nanoseconds=10):
    if not all(points[i][1] < points[i+1][1] for i in range(len(points) - 1)):
        raise Exception("Energy points are not sorted or contain duplicates")
    half_window = nanoseconds / 2
    
    mean_points = []
    interval_left_index = 0
    interval_right_index = 0
    sum_value = 0
    num_value = 0
    for point in points:
        for point_index in range(interval_left_index, len(points)):
            if points[point_index][1] < point[1] - half_window:
                interval_left_index += 1
                if not math.isnan(points[point_index][0]):
                    sum_value -= points[point_index][0]
                    num_value -= 1
            else:
                break
        for point_index in range(interval_right_index, len(points)):
            if points[point_index][1] <= point[1] + half_window:
                interval_right_index += 1
                if not math.isnan(points[point_index][0]):
                    sum_value += points[point_index][0]
                    num_value += 1
            else:
                break
        mean_value = sum_value / num_value
        mean_points.append((mean_value, point[1]))
    
    return mean_points

def get_threshold_mean_std(points, threshold, value_index=0, standard_error=False):
    if not all(points[i][1] < points[i+1][1] for i in range(len(points) - 1)):
        raise Exception("Energy points are not sorted or contain duplicates")
    
    for i, point in enumerate(points):
        if point[1] >= threshold:
            threshold_points = np.array(tuple(zip(*points[i:]))[value_index])
            threshold_points = threshold_points[~np.isnan(threshold_points)]
            if standard_error:
                sample_std = threshold_points.std(ddof=1)
                return (threshold_points.mean(), sample_std, sample_std / np.sqrt(threshold_points.shape[0]))
            else:
                return (threshold_points.mean(), threshold_points.std(ddof=1))
    
    raise Exception("There are no points greater than the nanosecond threshold")

def get_energy_plots(df, name_column, energy_column, controls_num, language="EN", mean_threshold_ns=50, mean_window_ns=10,
                     mean_line=True, show_without_smoothing=True, save_file=None):
    
    df_plot = df[~df[energy_column].isna()][[name_column, energy_column]]
    axes_num = df_plot.shape[0]
    
    fig, axs = md.get_subplots_template(axes_num, controls_num)
    
    if language == "EN":
        time_label = 'Time, ns'
        energy_units = 'kcal/mol'
    elif language == "RU":
        time_label = 'Время, нс'
        energy_units = 'ккал/моль'
    else:
        raise Exception("Unknown language")
    
    df_plot["mean_std"] = df_plot[energy_column].apply(lambda x: get_threshold_mean_std(x, mean_threshold_ns))
    df_plot["values, time"] = df_plot[energy_column].apply(lambda x: tuple(zip(*x))[:2])
    df_plot["window values, time"] = df_plot[energy_column].apply(lambda x: tuple(zip(*mean_window(x, mean_window_ns))))
    
    if show_without_smoothing:
        energy_max = df_plot["values, time"].apply(lambda x: max(x[0])).max()
        energy_min = df_plot["values, time"].apply(lambda x: min(x[0])).min()
    else:
        energy_max = df_plot["window values, time"].apply(lambda x: max(x[0])).max()
        energy_min = df_plot["window values, time"].apply(lambda x: min(x[0])).min()
    
    if energy_max - energy_min > 40:
        interval_len = 10
    elif energy_max - energy_min > 20:
        interval_len = 5
    else:
        interval_len = 2
        
    yticks_max = int((energy_max // interval_len) * interval_len)
    yticks_min = int((energy_min // interval_len + (energy_min % interval_len != 0)) * interval_len)
    yticks = list(range(yticks_max, yticks_min - interval_len, -interval_len))

    time_max = df_plot["values, time"].apply(lambda x: max(x[1])).max()
    time_min = df_plot["values, time"].apply(lambda x: min(x[1])).min()
    
    if time_max - time_min > 200:
        t_interval_len = 50
    elif time_max - time_min > 100:
        t_interval_len = 20
    else:
        t_interval_len = 10
    
    xticks_max = int((time_max // t_interval_len) * t_interval_len)
    xticks_min = int((time_min // t_interval_len + (time_min % t_interval_len != 0)) * t_interval_len)
    xticks = list(range(xticks_min, xticks_max + t_interval_len, t_interval_len))
    
    if xticks[0] - time_min >= t_interval_len / 2:
        xticks = [int(time_min)] + xticks
    else:
        xticks = [int(time_min)] + xticks[1:]
    if time_max - xticks[-1] >= t_interval_len * 4/5:
        xticks = xticks + [int(time_max)]
    else:
        xticks = xticks[:-1] + [int(time_max)]
    
    for i, (name, (values, time), (window_values, window_time), (mean, std)) in enumerate(df_plot[[name_column, "values, time", "window values, time", "mean_std"]].values):
        if show_without_smoothing:
            axs[i].plot(time, values)
        
        axs[i].plot(window_time, window_values, linewidth=2)
        
        if mean_line:
            axs[i].plot(time, [mean]*len(time), linestyle='--', color='green')
        
        if len(name) > 32:
            axs[i].set_title(name, fontdict={'fontsize': 14}, x=0.49, y=0.05, pad=0)
        else:
            axs[i].set_title(name, fontdict={'fontsize': 24}, x=0.49, y=0.05, pad=0)
        axs[i].set_yticks(yticks)
        axs[i].set_ylabel(energy_column + ', ' + energy_units)
        axs[i].set_xticks(xticks)
        axs[i].set_xlabel(time_label)
        axs[i].set_ylim([energy_min, energy_max])
        
        axs[i].text(0.97, 0.88, f'{mean:.1f} ± {std:.1f} ' + energy_units, style='italic', bbox={'facecolor': 'white', 'alpha': 0.8}, fontsize=20, horizontalalignment="right", transform=axs[i].transAxes)

    fig.tight_layout(pad=1)
    fig.subplots_adjust(wspace=0.03, hspace=0.05)
    if save_file is not None:
        fig.savefig(save_file, dpi=300)

def get_energy_bar(df, name_column, energy_column, controls_num, mean_threshold_ns, language="EN", save_file=None):
    df = df[~df[energy_column].isna()][[name_column, energy_column]]
        
    if language == "EN":
        energy_units = 'kcal/mol'
    elif language == "RU":
        energy_units = 'ккал/моль'
    else:
        raise Exception("Unknown language")

    mean_std_sem = df[energy_column].apply(lambda x: get_threshold_mean_std(x, mean_threshold_ns, standard_error=True)).apply(pd.Series)
    colors = ["c"]*controls_num + ["b"]*(df.shape[0] - controls_num)
    max_name_len = df[name_column].apply(len).max()
    
    fig = plt.figure(figsize=(16, 8))
    plt.bar(x=df[name_column], height=mean_std_sem[0], yerr=mean_std_sem[2], capsize=5, color=colors)
    plt.gca().invert_yaxis()
    if max_name_len < 8:
        plt.xticks(fontsize=26)
    elif max_name_len < 12:
        plt.xticks(fontsize=20)
    elif max_name_len < 16:
        plt.xticks(fontsize=16)
    else:
        plt.xticks(fontsize=16, rotation=30, ha="right")
    
    plt.yticks(fontsize=26)
    plt.ylabel(energy_column + ', ' + energy_units, fontsize=30)

    fig.tight_layout()
    if save_file:
        fig.savefig(save_file, dpi=300)