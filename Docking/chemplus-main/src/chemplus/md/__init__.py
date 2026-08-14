import numpy as np
import matplotlib.pyplot as plt

def get_time_range(frames_per_ns, start_end_interval, sub_start_end_interval=None):
    start, end, step = start_end_interval
    time_arr = np.arange(start, end+1, step) / frames_per_ns
    
    if sub_start_end_interval is not None:
        sub_start, sub_end, sub_interval = sub_start_end_interval
        return time_arr[sub_start-1:sub_end:sub_interval]
    else:
        return time_arr

def get_subplots_template(axes_num, controls_num, rms2d=False):
    if axes_num <= 3:
        ncols = axes_num
        nrows = 1
        empty_axes = ()
    elif axes_num == 4:
        ncols = 2
        nrows = 2
        empty_axes = ()
    elif axes_num == 5 and controls_num == 1:
        ncols = 2
        nrows = 3
        empty_axes = ((0, 1),)
    else:
        ncols = 3
        nrows = axes_num // ncols + (axes_num % ncols > 0)
        empty_axes_num = (ncols*nrows) % axes_num
        if empty_axes_num == 0:
            empty_axes = ()
        elif empty_axes_num == 1:
            if (controls_num % 3) == 2:
                empty_axes = ((controls_num // 3, 2),)
            else:
                empty_axes = ((nrows - 1, 2),)
        elif empty_axes_num == 2:
            if (controls_num % 3) == 1:
                empty_axes = ((0, 0), (0, 2))
            elif (controls_num % 3) == 2:
                empty_axes = ((controls_num // 3, 2), (nrows - 1, 2))
            else:
                empty_axes = ((nrows - 1, 1), (nrows - 1, 2))
    
    if not rms2d:
        fig, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=(ncols*5 + 1, nrows*3 + 1))
    else:
        fig, axs = plt.subplots(nrows, ncols + 1, squeeze=False, figsize=(ncols*5 + 2, nrows*5 + 1), 
                                gridspec_kw=dict(width_ratios=[1.0]*ncols + [0.1]))
    
    new_axes = []
    for i in range(nrows):
        for j in range(ncols):
            if (i, j) not in empty_axes:
                if j != 0 and (i, j - 1) not in empty_axes:
                    axs[i, j].set_yticklabels([])
                    axs[i, j].set_ylabel("Dummy", visible=False)
                if (rms2d and i != 0 and (i - 1, j) not in empty_axes) or (not rms2d and i != nrows - 1 and (i + 1, j) not in empty_axes):
                    axs[i, j].set_xticklabels([])
                    axs[i, j].set_xlabel("Dummy", visible=False)
                if rms2d:
                    axs[i, j].xaxis.tick_top()
                    axs[i, j].xaxis.set_label_position('top') 
                    axs[i, j].yaxis.tick_left()
                axs[i, j].tick_params(axis='both', which='major', labelsize=18)
                axs[i, j].yaxis.label.set_size(22)
                axs[i, j].xaxis.label.set_size(22)
                new_axes.append(axs[i, j])
            else:
                fig.delaxes(axs[i, j])
    
    if rms2d:
        for i in range(nrows):
            axs[i, ncols].tick_params(axis='both', which='major', labelsize=18) 
            axs[i, ncols].yaxis.label.set_size(22)
            axs[i, ncols].yaxis.label.set_rotation(270)
            axs[i, ncols].yaxis.labelpad = 27
            new_axes.append(axs[i, ncols])
    
    return fig, new_axes