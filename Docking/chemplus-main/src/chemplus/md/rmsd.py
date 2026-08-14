import os
import glob
import pandas as pd
import numpy as np
from chemplus import md
from io import StringIO
import seaborn as sns; sns.set_theme()

def rmsd_dat_to_stringio(dat_file):
    if not os.path.exists(dat_file):
        raise Exception("'" + dat_file + "' does not exists")
    
    dat_content = open(dat_file).readlines()
    csv_content_lines = [','.join(i.strip().split()) for i in dat_content]
    csv_content = '\n'.join(csv_content_lines).strip()
    
    return StringIO(csv_content)

def get_rmsd_df(rmsd_dir, frames_per_ns, start_end_interval, 
                rms2d=False, column_name=None, names_list=None):
    
    if not os.path.exists(rmsd_dir):
        raise Exception("RMSD directory does not exists")

    if column_name is not None:
        rmsd_column = column_name
    elif rms2d:
        rmsd_column = "RMS2D"
    else:
        rmsd_column = "RMSD"
    rmsd_time = md.get_time_range(frames_per_ns, start_end_interval)
    
    pd_list = []
    for rmsd_file in glob.glob(rmsd_dir + os.sep + "*.dat"):
        filename = os.path.basename(rmsd_file).split(".")[0]
        if names_list is not None and filename not in names_list:
            continue
        
        try:
            string_io = rmsd_dat_to_stringio(rmsd_file)
        except Exception as e:
            print("Skipping file:", e)
            continue
        
        if rms2d:
            rmsd_values = pd.read_csv(string_io).drop(columns='#Frame').to_numpy()
        else:
            df_rmsd = pd.read_csv(string_io, names=['frame', 'rmsd'], header=0)
            rmsd_values = df_rmsd['rmsd'].values
        if rmsd_values.shape[0] != rmsd_time.shape[0]:
            print(f"Skipping file: the number of rmsd values and frames is not equal in \"{filename}\".")
            continue
        
        pd_list.append({"Name" : filename, rmsd_column : (rmsd_values, rmsd_time)})
    
    return pd.DataFrame(pd_list)

def get_rmsd_plots(df, name_column, rmsd_column, controls_num, language="EN", rmsd_max=None, save_file=None):
    
    df_plot = df[~df[rmsd_column].isna()][[name_column, rmsd_column]]
    axes_num = df_plot.shape[0]
    
    fig, axs = md.get_subplots_template(axes_num, controls_num)
    
    if language == "EN":
        time_label = 'Time, ns'
    elif language == "RU":
        time_label = 'Время, нс'
    else:
        raise Exception("Unknown language")
    
    df_plot["mean_std"] = df_plot[rmsd_column].apply(lambda x: (x[0].mean(), x[0].std()))
    
    if rmsd_max is None:
        rmsd_max = df_plot[rmsd_column].apply(lambda x: max(x[0])).max()
    
    if rmsd_max >= 8:
        interval_len = 2
    elif rmsd_max >= 4:
        interval_len = 1
    else:
        interval_len = 0.5
        
    yticks_max = int((rmsd_max // interval_len) * interval_len)
    yticks_min = 0
    yticks = list(np.arange(yticks_min, yticks_max + interval_len, interval_len))

    time_max = df_plot[rmsd_column].apply(lambda x: max(x[1])).max()
    time_min = df_plot[rmsd_column].apply(lambda x: min(x[1])).min()
    
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
    
    for i, (name, (values, time), (mean, std)) in enumerate(df_plot[[name_column, rmsd_column, "mean_std"]].values):
        axs[i].plot(time, values)
        
        if len(name) > 32:
            axs[i].set_title(name, fontdict={'fontsize': 14}, x=0.49, y=0.05, pad=0)
        else:
            axs[i].set_title(name, fontdict={'fontsize': 24}, x=0.49, y=0.05, pad=0)
        axs[i].set_yticks(yticks)
        axs[i].set_ylabel("RMSD, Å")
        axs[i].set_xticks(xticks)
        axs[i].set_xlabel(time_label)
        axs[i].set_ylim([0, rmsd_max])
        
        axs[i].text(0.97, 0.88, f'{mean:.1f} ± {std:.1f} Å', style='italic', bbox={'facecolor': 'white', 'alpha': 0.8}, fontsize=20, horizontalalignment="right", transform=axs[i].transAxes)

    fig.tight_layout(pad=1)
    fig.subplots_adjust(wspace=0.03, hspace=0.1)
    if save_file is not None:
        fig.savefig(save_file, dpi=300)

def get_rms2d_plots(df, name_column, rmsd_column, controls_num, language="EN", rmsd_max=None, save_file=None):
    #TODO time arrays equality, etc.
    df_plot = df[~df[rmsd_column].isna()][[name_column, rmsd_column]]
    axes_num = df_plot.shape[0]
        
    shape_max = df_plot[rmsd_column].apply(lambda x: x[1].shape[0]).max()
    shape_min = df_plot[rmsd_column].apply(lambda x: x[1].shape[0]).min()
    
    if shape_max != shape_min:
        raise Exception("RMS2D different dimensions")
    else:
        rms2d_shape = shape_max
    
    fig, axs = md.get_subplots_template(axes_num, controls_num, rms2d=True)
    
    if language == "EN":
        time_label = 'Time, ns'
    elif language == "RU":
        time_label = 'Время, нс'
    else:
        raise Exception("Unknown language")
    
    df_plot["mean_std"] = df_plot[rmsd_column].apply(lambda x: (np.mean(x[0]), np.std(x[0])))
    
    if rmsd_max is None:
        rmsd_max = df_plot[rmsd_column].apply(lambda x: np.max(x[0])).max()
    
    if rmsd_max > 8:
        interval_len = 2
    elif rmsd_max > 4:
        interval_len = 1
    else:
        interval_len = 0.5
        
    yticks_max = int((rmsd_max // interval_len) * interval_len)
    yticks_min = 0
    yticks = list(np.arange(yticks_min, yticks_max + interval_len, interval_len))

    time_max = df_plot[rmsd_column].apply(lambda x: max(x[1])).max()
    time_min = df_plot[rmsd_column].apply(lambda x: min(x[1])).min()
    
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

    #p = (n-1) * (x - xmin) / (xmax - xmin) + 1
    xticks_positions = [(rms2d_shape - 1) * (tick - time_min) / (time_max - time_min) + 1 for tick in xticks]
    
    for i, (name, (values, time), (mean, std)) in enumerate(df_plot[[name_column, rmsd_column, "mean_std"]].values):        
        sns.heatmap(values, ax=axs[i], cbar=False, xticklabels=False, yticklabels=False, vmax=rmsd_max)
        
        if len(name) > 32:
            axs[i].set_title(name, fontdict={'fontsize': 14}, x=0.49, y=-0.07, pad=0)
        else:
            axs[i].set_title(name, fontdict={'fontsize': 24}, x=0.49, y=-0.07, pad=0)
        if axs[i].xaxis.label.get_visible():
            axs[i].set_xticks(xticks_positions, labels=xticks)
            axs[i].set_xlabel(time_label)
        if axs[i].yaxis.label.get_visible():
            axs[i].set_yticks(xticks_positions, labels=xticks)
            axs[i].set_ylabel(time_label)
        
    for i in range(df_plot.shape[0], len(axs)):
        fig.colorbar(axs[0].collections[0], cax=axs[i])
        axs[i].set_ylabel('RMSD, Å')
    
    fig.tight_layout(pad=1)
    fig.subplots_adjust(wspace=0.1, hspace=0.12)
    if save_file is not None:
        fig.savefig(save_file, dpi=300)
