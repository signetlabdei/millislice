import sem
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import seaborn as sns
import pandas as pd
# from pympler import tracker

# Functions

def plot_forall_static(static, param_ca, param_no_ca, versus, fewer_images=False):
    # tr = tracker.SummaryTracker()
    # Get list of values of params that are supposed to be static
    campaign = sem.CampaignManager.load('./slicing-res')
    loaded_params = campaign.db.get_all_values_of_all_params()
    # For now, just one..
    static_values = loaded_params[static]
    counter = 0
    for val in static_values:
        counter = counter + 1
        # Restrict params
        param_ca[static] = val
        param_no_ca[static] = val
        # Ugly, but lazy
        metric_bucket_dummy = {'versus':val}
        static_formatted = sanitize_versus(vs=static, metric_bucket=metric_bucket_dummy)
        val_formatted = metric_bucket_dummy['versus']
        # Save the plot
        out_dir = f"./slicing-plots/versus_{versus}/{val_formatted}/"
        os.makedirs(out_dir, exist_ok=True)
        # Use lower level functions to plot
        fig = plot_all_metrics(param_no_ca=param_no_ca, param_ca=param_ca, versus=versus,
                            fewer_images=fewer_images, top_path=out_dir)

        fig.suptitle(f"System performance for {static_formatted} = {val_formatted}", fontsize=16)
        plt.savefig(f"{out_dir}System_performance.png" )
        plt.close('fig')
        del fig
        print(f"{counter/len(static_values)*100} % done!")
        #tr.print_diff()

def plot_all_metrics(param_ca, param_no_ca, versus=None, fewer_images=False, top_path=None):

    fig, ax = plt.subplots(constrained_layout=True, nrows=2, ncols=3)
    campaign = sem.CampaignManager.load('./slicing-res')
    trace_str_rx_pckt = 'test_RxPacketTrace.txt' # Always same name

    for prot in ['URLLC', 'eMBB']:
        thr = []
        delay = []
        loss = []

        # Select proper trace file
        if prot == 'URLLC':
            trace_str_dl = 'test_urllc-dl-app-trace.txt'
            trace_str_ul = 'test_urllc-ul-sink-app-trace.txt'
            sub_col = 0
        elif prot == 'eMBB':
            trace_str_dl = 'test_eMBB-dl-app-trace.txt'
            trace_str_ul = 'test_eMBB-ul-app-trace.txt'
            sub_col = 1

        for param in param_ca, param_no_ca:
            print(f"{prot} stats, params {param}")
            # Load the desired datasets
            # Load results, specify params if given on input
            # Get the required files IDs
            if param is not None:
                res_data = campaign.db.get_results(param)
            else:
                res_data = campaign.db.get_results()
            for res_istance in res_data:
                res_id = res_istance['meta']['id']
                # Load all the desired traces
                dl_path = campaign.db.get_result_files(res_id)[trace_str_dl]
                ul_path = campaign.db.get_result_files(res_id)[trace_str_ul]
                # Save both results and relative params
                dl_df = pd.read_csv(filepath_or_buffer=dl_path, header=0, delimiter='\t', low_memory=True)
                ul_df = pd.read_csv(filepath_or_buffer=ul_path, header=0, delimiter='\t', low_memory=True)

                # Keep just some of the params

                # Improve data structure, keep just relevant data
                ul_df = sanitize_dataframe(ul_df, res_istance['params']['maxStart']*1e9) # sec to ns ns in the traces
                dl_df = sanitize_dataframe(dl_df, res_istance['params']['maxStart']*1e9) # sec to ns ns in the traces

                # Compute metrics here
                loss.append({'mean': pkt_loss_app(dl_df, ul_df), 'params': res_istance['params']})

                thr.append({'mean':throughput_app(dl_df, bearer_type=prot, params=res_istance['params']), 
                            'params': res_istance['params']})

                delay.append({'mean':delay_app(dl_df), 'params': res_istance['params']})
                
        # Plot the various metrics 
        info = {'prot':prot, 'metric':'Delay', 'unit':'[ms]'}       
        plot_lines_versus(metric_bucket=delay, info=info, s_path=top_path, versus=versus, fig=fig, ax=ax[sub_col, 2])
        info = {'prot':prot, 'metric':'Throughput', 'unit':'[Mbit/s]'}
        plot_lines_versus(metric_bucket=thr, info=info, s_path=top_path, versus=versus, fig=fig, ax=ax[sub_col, 1])
        info = {'prot':prot, 'metric':'Packet loss', 'unit':''}
        plot_lines_versus(loss, s_path=top_path, info=info, versus=versus, fig=fig, ax=ax[sub_col, 0])

    return fig

def group_cc_strat(metric_frame):
    metric_frame['mode'] =  metric_frame['mode'].replace(1, 'no CA, ')
    metric_frame['mode'] =  metric_frame['mode'].replace(2, 'CA, ')
    metric_frame['ccMan'] =  metric_frame['ccMan'].replace(0, 'SplitDrb')
    metric_frame['ccMan'] =  metric_frame['ccMan'].replace(1, 'SlicingDrb')

    metric_frame['CC strategy'] = metric_frame['mode'] + metric_frame['ccMan']
    metric_frame['CC strategy'] =  metric_frame['CC strategy'].replace('no CA, SlicingDrb', 'no CA')

    return metric_frame

def plot_lines_versus(metric_bucket, info, s_path, versus, fig=None, ax=None):

    dummy_ax = ax
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
        sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})

    metric_data = []
    mode_data = []
    ccman_data = []
    versus_data = []

    for res in metric_bucket:
        metric_data.append(res['mean'])
        mode_data.append(res['params']['mode'])
        ccman_data.append(res['params']['ccMan'])
        versus_data.append(res['params'][versus])

    frame = {
        'metric': metric_data,
        'mode': mode_data,
        'versus': versus_data,
        'ccMan': ccman_data
    }
    metric_frame = pd.DataFrame(data=frame)

    metric_frame = group_cc_strat(metric_frame)
    filename = f"{info['prot']}_{info['metric']}_vs{versus}.png"
    temp = sanitize_versus(metric_bucket=metric_frame, vs=versus)
    if temp is not None:
        versus = temp

    g = sns.lineplot(data=metric_frame, x='versus', y='metric', err_style='bars', hue='CC strategy', ax=ax)

    # Set graphical properties, title and filename
    ax.set_ylabel(f"{info['metric']} {info['unit']} \n", fontsize=12)
    ax.set_xlabel(f"{versus}", fontsize=12)
    plot_title = f"{info['metric']} {info['prot']} vs. {versus}"

    if dummy_ax is None:
        fig.set_size_inches(count_amount_uniques(versus_data)*2, 8)    
        fig.suptitle(f"{plot_tile} \n", fontsize=12)
        # Save, create dir if doesn't exist 
        out_dir = f"./slicing-plots/{s_path}/"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f"{out_dir}{filename}")
        plt.close('fig')
    else:
        fig.set_size_inches(count_amount_uniques(versus_data)*2*3, 10)
        ax.grid(color='#b3b3b3')
        ax.set_facecolor('#f5f5fa')
        ax.title.set_text(f"{plot_title} \n")
        for spine in ax.spines.values():
            spine.set_edgecolor('#b3b3b3')

    del dummy_ax
    del g
    #Ylim
    #g.set(ylim=(0, None)) 
'''
def plot_line(metric_frame, metric, title, s_path, fname, overlays=None):

    fig, ax = plt.subplots(constrained_layout=True)
    sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})
    color = '#7a4e4f'

    sns.lineplot(data=metric_frame, x='x', y='y', color=color)

    x_handles = []
    y_handles = []

    # Plot overlays if available
    if overlays is not None:
        for item in overlays['x']:
            ax.axvline(x=item, color=[0,0,0], linewidth=0.7, label='appEnd')
        for item in overlays['y']:
            ax.axhline(y=item, color=color, linestyle='dashed', linewidth=1, label='Average value of whole system')

    #f, axes = plt.subplots(1, 2)
    plt.ylabel(f"{metric} \n", fontsize=11)
    plt.title(title + '\n') 

    ax.legend(loc='best')

    out_dir = s_path
    os.makedirs(out_dir, exist_ok=True)
    fig.set_size_inches(8, 6)

    plt.savefig(out_dir + fname)
    plt.close('fig')
'''
def plot_distr_bins(metric_frame, metric, title, s_path):
    # Make sure figure is clean
    fig, ax = plt.subplots(constrained_layout=True)
    sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})

    sns.distplot(metric_frame, kde=False, norm_hist=True)

    plt.xlabel(f"{metric} \n", fontsize=11)
    plt.title(title + '\n') 

    fig.set_size_inches(8, 6)

    # Save, create dir if doesn't exist       
    out_dir = f"{s_path}detailed/"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}{metric}")

    plt.close('fig')

def plot_metric_box(metric_frame, metric, title, s_path, versus):
    # Make sure figure is clean
    fig, ax = plt.subplots(constrained_layout=True)

    # Clean up and group by ccMan strategy
    metric_frame = group_cc_strat(metric_frame)
    x_label = sanitize_versus(vs=versus, metric_bucket=metric_frame)

    light_palette = ['#90a5e0', '#90a5e0', '#c27a7c']
    dark_palette = ['#465782','#465782', '#7a4e4f']
    sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})

    # Plot sum as background
    metric_frame['band_alloc_cc1'] = metric_frame['band_alloc_cc1'] + metric_frame['band_alloc_cc0']
    sns.barplot(x='versus', y='band_alloc_cc1', hue='CC strategy', data=metric_frame, palette=sns.color_palette('pastel'))
    # Plot cc0 on foreground
    ax_bckg = sns.barplot(x='versus', y='band_alloc_cc0', hue='CC strategy', data=metric_frame, palette=sns.color_palette('muted'))
    handles, labels = ax_bckg.get_legend_handles_labels()
    for dummy in range(0, 3):
        labels[dummy] = 'CC1 - ' +  labels[dummy] 
        labels[3 + dummy] = 'CC0 - ' +  labels[3 + dummy]
    ax_bckg.legend(handles=handles[1:], labels=labels[1:], ncol=2, loc='best')

    # ax_bckg.legend(handles=handles[3:], labels=labels[3:])
    
    # Title, labels ecc.
    fig.set_size_inches(6, 8.5)
    filename = f"{metric}.png"
    plt.ylabel(f"{metric} \n", fontsize=12)
    plt.xlabel(f"{x_label}", fontsize=12)
    #ax.set_xlabel('')
    plt.title(f"{title} \n")

    # Save, create dir if doesn't exist       
    out_dir = s_path
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir + filename)

    plt.close(fig)

def plot_metrics_generic(metric_bucket, metric, prot, s_path, unit, vs=None):
    """ 
    Plots metric mean, CI and all run samples
    Args:
        versus (str): param to use on the x axis
    """
    # Make sure figure is clean
    plt.clf()

    # Build dataframe
    metric_data = []
    mode_data = []
    versus_data = []

    for res in metric_bucket:
        metric_data.append(res['mean'])
        mode_data.append(res['params']['mode'])
        versus_data.append(res['params'][vs])

    frame = {
        'metric': metric_data,
        'mode': mode_data,
        'versus': versus_data
    }

    metric_frame = pd.DataFrame(data=frame)
    metric_frame['mode'].replace(1, 'no CA',inplace=True)
    metric_frame['mode'].replace(2, 'CA', inplace=True)
    
    # Colors
    # fig, (box_ax, viol_ax) = plt.subplots(2, 1, constrained_layout=True, sharex=True, sharey=True)
    fig, ax = plt.subplots(constrained_layout=True)
    # Avoid having strange ticks formatting
    plt.ticklabel_format(useOffset=False, style='plain')

    dark_palette = ['#465782', '#7a4e4f']
    light_palette = ['#90a5e0', '#c27a7c']
    sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})

      # Save, with the proper size
    if check_constant(versus_data):
        filename = f"{prot}_{metric}_CA_vs_nonCA.png"
        plot_title = f"{prot} {metric}"
    else:
        filename = f"{prot}_{metric}_vs{vs}_CA_vs_nonCA.png"
        # viol_ax.set_xlabel(f"{vs}", fontsize=11)
        # box_ax.set_xlabel(f"{vs}", fontsize=11)


    # Violin plotax
    # sns.violinplot(data=metric_frame, y='metric', x='versus', hue='mode', palette=light_palette, split=True, inner='stick', ax=viol_ax)

    vs = sanitize_versus(metric_bucket=metric_frame, vs=vs)
    # Boxplot
    sns.boxplot(data=metric_frame, y='metric', x='versus', hue='mode', palette=light_palette, ax=ax)
    strip_handle = sns.stripplot(x="versus", y="metric", hue="mode", data=metric_frame, dodge=True, palette=dark_palette, ax=ax)

    # Remove legend duplicate
    handles, labels = strip_handle.get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2])

    # Save, with the proper size
    if check_constant(versus_data):
         # Overlay the mean values
        # overlay_means(metric_bucket, palette=dark_palette, vs=vs, vs_data=versus_data)

        # Set graphical properties
        fig.set_size_inches(4, 8)
        # Set title and filename
        plot_title = f"{prot} {metric}"
    else:
        # Overlay the mean values, if traces are not empty
        # overlay_means(metric_bucket, palette=dark_palette, vs=vs, vs_data=versus_data)

        # Set graphical properties
        fig.set_size_inches(count_amount_uniques(versus_data)*2.5, 7.5)
         # Set title and filename
        plot_title = f"{prot} {metric} vs. {vs}"
        ax.set_xlabel(f"{vs}", fontsize=12)
        # viol_ax.set_xlabel(f"{vs}", fontsize=11)
        # box_ax.set_xlabel(f"{vs}", fontsize=11)
        

    ax.set_ylabel(f"{metric} {unit} \n", fontsize=12)
    fig.suptitle(plot_title + '\n', fontsize=12)

    # Save, create dir if doesn't exist       
    out_dir = f"./slicing-plots/{s_path}/"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir + filename)

    plt.close('fig')
'''
def overlay_means(metric_bucket, palette, vs, vs_data):

   # Overlay mean value
    # Obtain info regarding bounds
    left_orig, right_orig = plt.xlim()
    # Compute means
    means_bucket = compute_means(group_by_params(metric_bucket))
    # Get possible, different vs values
    vs_uniques = list(set(vs_data))
    vs_uniques.sort()

    for index in range(len(vs_uniques)):
        just_right_means = find_elements(means_bucket, vs, vs_uniques[index])
        # If no trace loaded for such param combination, do nothing
        if(len(just_right_means) >= 2):
            no_ca_mean = find_elements(just_right_means, 'mode', 1)[0]['mean']
            ca_mean = find_elements(just_right_means, 'mode', 2)[0]['mean']
            # Get plot limits and plot
            width = (right_orig - left_orig)/len(vs_uniques)
            left = left_orig + index*width
            right = left + width
            plt.plot([left, right], [no_ca_mean, no_ca_mean], color=palette[0], linestyle='--', linewidth=1)
            plt.plot([left, right], [ca_mean, ca_mean], color=palette[1], linestyle='--', linewidth=1)
'''

def sanitize_versus(vs, metric_bucket):
    if(vs == 'embbUdpIPI'):
        metric_bucket['versus'] = round(1024*8/(metric_bucket['versus']), 1) # packet_size*bits in a bye/rate
        return 'eMBB sources rate [Mbit/s]'  
    if(vs == 'urllcUdpIPI'):
        metric_bucket['versus'] = round(1024*8/(metric_bucket['versus']), 1) # packet_size*bits in a bye/rate
        return 'URLLC sources rate [Mbit/s]'
    if(vs == 'ccRatio'):
        return 'Ratio of bw allocated to CC0'

def find_elements(bucket, param, value):
    out = []
    for element in bucket:
        temp = element['params'][param]
        if temp == value:
            out.append(element)
    return out

def group_by_params(metric_bucket):
    # Remove param specifying different runs
    for sim in metric_bucket:
        sim['params'].pop('RngRun', None)
        sim['params'].pop('runSet', None)
    out_bucket = []
    # Group sims having same param
    while(len(metric_bucket) > 0):
        temp_param = metric_bucket[0]['params']
        mean_bucket = []
        waste_bucket = []
        # Find sims with same params
        for sim in metric_bucket:
            if sim['params'] == temp_param:
                mean_bucket.append(sim['mean'])
                waste_bucket.append(sim)
        # Create entry for same params, different runs
        out_bucket.append({
            'mean': mean_bucket,
            'params': temp_param
        })
        # Remove elements that have been joined
        for sim in waste_bucket:
            metric_bucket.remove(sim)

    return out_bucket

def load_results(trace_name, param=None):
    """ 
    Loads the results from specifics file traces.
    If params are specified, it loads just the results of the
    simulations corresponding to such mean.

    Args:
        trace_name (str): filename of the trace(s) to load

    Returns:
        A list of results matching the query, one entry for every simulation run. 
        The structure of the returning value is a list, comprised of dictionaries
        having keys 'results' and 'params'. The first holds an array of results     
        for every traced parameter, the latter the combination of params that 
        originated such outputs.
    """



    # Get list containing data of the trace for the various param combination
    # and combination of params that generated it
    res_bucket = []
    
    return res_bucket

def sanitize_dataframe(dataframe, treshold):
    # Remove trailing whitespaces from cols names
    dataframe = dataframe.rename(columns=lambda x: x.strip())
    # We want to keep trace just of packets transmitted after all apps started
    if 'tx_time' in dataframe.columns:
        dataframe = dataframe[dataframe['tx_time'] > treshold]   
    else:
        dataframe = dataframe[dataframe['time'] > treshold/1e9] # Need secs here

    # Remove unused cols
    if 'TBler' in dataframe.columns:
        dataframe.drop(columns=['TBler', 'corrupt', 'mcs', 'rv' , '1stSym', 'DL/UL' , 'time', 'tbSize'], inplace=True)

    return dataframe

def sinr_overall(trace_data):

    print('--Computing SINR statistics--')

    # Start by plotiing SINR distribution, averaged over the runs
    sinr_bucket = []

    for item in trace_data:
        sinr_bucket.extend(item['results']['SINR(dB)'])

    snr_frame = {
        'sinr': sinr_bucket
    }

    return pd.DataFrame(data=snr_frame)


def band_allocation(trace_data, versus):

    print('--Computing band allocation metric--')

    ccMan_data = []
    mode_data = []
    cc0_data = []
    cc1_data = []
    versus_data = []

    for item in trace_data:
        # Find out total amount symbols available
        avail_sym = item['results']['frame'].iloc[-1] - item['results']['frame'].iloc[0]
        avail_sym = avail_sym*10*22 # Frames*subframes in a frame*symbols in a subframe
        # Get info regarding first CC
        item_cc0 = item['results'][item['results']['ccId'] == 0]
        used_sym_cc0 = item_cc0['symbol#'].sum()

        versus_data.append(item['params'][versus])
        mode_data.append(item['params']['mode'])
        ccMan_data.append(item['params']['ccMan'])
        # If we are using CA, get also info regarding secondary CC
        if item['params']['mode'] == 2:
            item_cc1 = item['results'][item['results']['ccId'] == 1]
            used_sym_cc1 = item_cc1['symbol#'].sum()
            # Normalize usage for stacked 
            cc0_data.append((used_sym_cc0/avail_sym)*item['params']['ccRatio'])
            cc1_data.append((used_sym_cc1/avail_sym)*(1-item['params']['ccRatio']))
            
        else:
            cc1_data.append(0)
            cc0_data.append(used_sym_cc0/avail_sym) 

    frame = {
        'band_alloc_cc0': cc0_data,
        'band_alloc_cc1': cc1_data,
        'ccMan': ccMan_data,
        'mode': mode_data,
        'versus': versus_data
    }

    return pd.DataFrame(data=frame)

def throughput_app_det(trace_data, bearer_type, vs, s_path):

    out = []
    versus_data = []
    # Compute and plot overall throughput in the worst case
    
    for sim in trace_data:
        versus_data.append(sim['params'][vs])

    # Obtain unique values
    versus_data = list(set(versus_data))

    for vs_value in versus_data:
        # Keep just traces using such value of the vs param
        temp_trace_data = []
        for item in trace_data:
            if (item['params'][vs] == vs_value):
                temp_trace_data.append(item)

        # Compute overall mean throughputs, pick the one with the worst average across the runs
        thr_bucket = throughput_app(temp_trace_data, bearer_type)
        mean_frame = []
        for item in thr_bucket:
            mean_frame.append(item['mean'])

        temp_frame = pd.DataFrame({
            'value': mean_frame
        })

        choosen_one = copy.deepcopy(trace_data[temp_frame['value'].idxmin()]['results'])
        # Compute throughput versus time
        choosen_one['rx_time'] = (choosen_one['rx_time']/10e8).round(2)
        packets_rx = choosen_one.groupby(['rx_time']).count()

        # Fix col names
        packets_rx['x'] = packets_rx.index.values
        packets_rx.rename(columns={'tx_time':'y'}, inplace=True)
        # Packets every 0.1s --> Mbit/s
        packets_rx['y'] = packets_rx['y']*1024*8/1e5 # 10e6 for bit to Mbit, then 0.1 to 1 s
        packets_rx.drop(columns=['pkt_size', 'seq_num', 'node_id'], inplace=True)

        temp_title = 'Throughput of whole system vs time [s]'
        temp_path =  s_path + 'detailed/'
        filename = f"{bearer_type}_{vs}_{vs_value}"

        # Overlay mean throughput and appEnd 
        overlays = {
            'x': [temp_trace_data[0]['params']['appEnd']],
            'y': [temp_frame['value'].min()]
        }

        plot_line(metric_frame=packets_rx, metric='Throughput [Mbit/s]', title=temp_title, 
                    s_path=temp_path, fname=filename, overlays=overlays)

    return out

def throughput_app(trace_data, bearer_type, params):
    """ 
    Computes the average throughput @ APP layer
    If parameters combination are provided, then only the simulations for
    such mean are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """

    g = (len(trace_data.index)*1024*8)/((params['appEnd'] -
                                        params['maxStart'])*1e6)  # computing overall throughput
    # computing per user throughput
    if bearer_type == 'URLLC':
        single_g = g/(params['numUrllcUes'])
    else:
        single_g = g/(params['numEmbbUes'])

    return single_g

def delay_app(trace_data):
    """ 
    Computes the average delay @ APP layer.
    If parameters combination are provided, then only the simulations for
    such mean are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """
    time_rx = trace_data['rx_time']
    # get time of tx
    time_tx = trace_data['tx_time']
    # packet delay
    #'var': pck_delay.std(),  # Output both latency and jitter
    pck_delay = (time_rx - time_tx)/1e6
    return pck_delay.mean()

def pkt_loss_app(trace_dl, trace_ul):
    """ 
    Computes the average delay @ APP layer.
    If parameters combination are provided, then only the simulations for
    such mean are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """
    loss = 0
    sent = len(trace_ul.index)
    # Overall lost packets
    dropped = sent - len(trace_dl.index)
    
    return dropped/len(trace_ul.index)

# Small, support functions
def check_constant(bucket):
    return bucket[1:] == bucket[:-1]

def count_amount_uniques(bucket):
    return len(set(bucket))

def compute_means(metric_bucket):
    # Save original data
    out_bucket = copy.deepcopy(metric_bucket)

    for index in range(len(metric_bucket)):
        out_bucket[index]['mean'] = mean(metric_bucket[index]['mean'])
    return out_bucket

# Actual metrics computation
# Try plot
print('CA using f0=28GHz, f1=10Ghz; non CA using f0=28GhzL: vs eMBB rates')
ca_params = {'f0': 28e9, 'f1':10e9,'mode': 2}
no_ca_params = {'f0': 28e9, 'mode': 1}

print('Computing stats')
plot_forall_static(param_ca=ca_params, param_no_ca=no_ca_params, versus='embbUdpIPI', fewer_images=True, static='urllcUdpIPI') 

print('CA using f0=28GHz, f1=10Ghz; non CA using f0=28GhzL: vs URLLC rates')
ca_params = {'f0': 28e9, 'f1':10e9, 'mode': 2}
no_ca_params = {'f0': 28e9, 'mode': 1}

print('Computing stats')
plot_forall_static(param_ca=ca_params, param_no_ca=no_ca_params, versus='urllcUdpIPI', fewer_images=True, static='embbUdpIPI')
