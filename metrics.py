import sem
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import seaborn as sns
import pandas as pd


# Functions
"""
def print_metric(metric_bucket, intro, just_mean=0):

    print(intro)
    # Find out which param is changing, if not remove it
    params_to_rem = []
    params_list = list(metric_bucket[0]['params'].keys())   # list of params
    for item in params_list:
        # Check if that param is the same for all simulations
        temp_bucket = []
        for sim in metric_bucket:
            temp_bucket.append(sim['params'][item])
        if (check_constant(temp_bucket)):
            params_to_rem.append(item)
    # Remove all constant params
    for sim in metric_bucket:
        for param in params_to_rem:
            sim['params'].pop(param, None)
    out = group_by_params(metric_bucket)

    if(just_mean == 0):
        print(out)
    else:
        print(compute_means(out))
    return out
"""
def plot_all_metrics(prot, param_ca=None, param_no_ca=None, versus=None):

    # Load the desired datasets
    # Select proper trace file
    if prot == 'urllc':
        trace_str_dl = 'test_urllc-dl-app-trace.txt'
        trace_str_ul = 'test_urllc-ul-sink-app-trace.txt'
        # Upper case for plot legend
        prot = 'URLLC'
    else:
        trace_str_dl = 'test_eMBB-dl-app-trace.txt'
        trace_str_ul = 'test_eMBB-ul-app-trace.txt'
        # Upper case for plot legend
        prot = 'eMBB'

    trace_str_rx_pckt = 'test_RxPacketTrace.txt' # Always same name
    
    sub_path = ''

    # Load results, specify params if given on input
    if param_no_ca is not None:
        trace_no_ca_dl = load_results(trace_name=trace_str_dl, param=param_no_ca)
        trace_no_ca_ul = load_results(trace_name=trace_str_ul, param=param_no_ca)
        trace_no_ca_rx_pckt = load_results(trace_name=trace_str_rx_pckt, param=param_no_ca)
        # Subfoloder name
        sub_path += print_dict(param_no_ca)
        if param_ca is not None and param_ca is not param_no_ca:
            trace_ca_dl = load_results(trace_name=trace_str_dl, param=param_ca)
            trace_ca_ul = load_results(trace_name=trace_str_ul, param=param_ca)
            trace_ca_rx_pckt = load_results(trace_name=trace_str_rx_pckt, param=param_ca)
            # Combine the traces lists:
            trace_dl = trace_no_ca_dl + trace_ca_dl
            trace_ul = trace_no_ca_ul + trace_ca_ul
            trace_rx_pckt = trace_no_ca_rx_pckt + trace_ca_rx_pckt
            sub_path += 'versus_' + print_dict(param_ca)
        else:
            trace_dl = trace_no_ca_dl 
            trace_ul = trace_no_ca_ul 
            trace_rx_pckt = trace_no_ca_rx_pckt   

    else:
        trace_dl = load_results(trace_name=trace_str_dl)
        trace_ul = load_results(trace_name=trace_str_ul)
        trace_rx_pckt = load_results(trace_name=trace_str_rx_pckt)
        sub_path = 'no_spec_params'


    # Call lower level function
    plot_distr_bins(metric_frame=sinr_overall(trace_rx_pckt), metric='SINR(dB)', title='Distribution of the SINR of all users, for all simulation runs', s_path=sub_path)
    throughput_app_det(trace_dl, prot, versus, s_path=sub_path)
    plot_metric_box(band_allocation(trace_rx_pckt), s_path=sub_path, metric='Band allocation', title='Band allocation metric')
    plot_metrics_generic(metric_bucket=pkt_loss_app(trace_dl, trace_ul), metric='Packet loss', prot=prot, s_path=sub_path, unit='', vs=versus)
    plot_metrics_generic(metric_bucket=throughput_app(trace_dl, bearer_type=prot), metric='Throughput', s_path=sub_path, prot=prot, unit='[Mbit/s]', vs=versus)
    plot_metrics_generic(metric_bucket=delay_app(trace_dl),  metric='Delay', prot=prot, s_path=sub_path, unit='[ms]', vs=versus)

def print_dict(param_dict):
    out = ''
    for key in param_dict.keys():
        out += f"{key}_"
        if isinstance(param_dict[key], float):
            temp = '{:.2e}'.format(param_dict[key])
            out += f"{temp}_"
        else:
            out += f"{param_dict[key]}_"

    return out

def plot_line(metric_frame, metric, title, s_path, overlays=None):

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

    out_dir = f"./slicing-plots/{s_path}"
    # os.makedirs(out_dir, exist_ok=True)
    fig.set_size_inches(8, 6)

    plt.savefig(out_dir)
    plt.close(fig)

def plot_distr_bins(metric_frame, metric, title, s_path):
    # Make sure figure is clean
    fig, ax = plt.subplots(constrained_layout=True)

    sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})

    color = '#7a4e4f'
    sns.distplot(metric_frame, kde=False, color=color, norm_hist=True)

    plt.xlabel(f"{metric} \n", fontsize=11)
    plt.title(title + '\n') 

    fig.set_size_inches(8, 6)

    # Save, create dir if doesn't exist       
    out_dir = f"./slicing-plots/{s_path}/detailed/"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir + metric)

    plt.close(fig)

def plot_metric_box(metric_bucket, metric, title, s_path):
    # Make sure figure is clean
    fig, ax = plt.subplots(constrained_layout=True)
    # Build dataframe
    values = []
    labels = []

    for res in metric_bucket:
        # Add in stats for primary CC
        values.append(res['cc0'])
        if res['params']['mode'] == 1: # If we are not using CA    
            labels.append('CC0, no CA')
        else:
            labels.append('CC0, CA')
            # Add in stats for secondary CC
            values.append(res['cc1'])
            labels.append('CC1, CA')

    frame = {
        'values': values,
        'labels': labels
    }

    metric_frame = pd.DataFrame(data=frame)
    # Actual plot
    

    light_palette = ['#90a5e0', '#90a5e0', '#c27a7c']
    dark_palette = ['#465782','#465782', '#7a4e4f']
    sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})
    ax = sns.boxplot(data=metric_frame, y='values', x='labels', palette=light_palette)
    ax = sns.stripplot(data=metric_frame, y='values', x='labels', palette=dark_palette)

    # Title, labels ecc.
    fig.set_size_inches(5, 7)
    filename = f"{metric}_CA_vs_nonCA.png"
    plt.ylabel(f"{metric} \n", fontsize=12)
    ax.set_xlabel('')
    plt.title(title + '\n')

    # Save, create dir if doesn't exist       
    out_dir = f"./slicing-plots/{s_path}/"
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
    metric_frame['mode'] =  metric_frame['mode'].replace(1, 'no CA')
    metric_frame['mode'] =  metric_frame['mode'].replace(2, 'CA')
    
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


def sanitize_versus(metric_bucket, vs):
    if(vs == 'embbUdpIPI'):
        metric_bucket['versus'] = round(1024*8/(metric_bucket['versus'])) # packet_size*bits in a bye/rate
        return 'UDP sources datarate [Mbit/s]'  

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

    # Get the required files IDs
    campaign = sem.CampaignManager.load('./slicing-res')
    if param is not None:
        res_data = campaign.db.get_results(param)
    else:
        res_data = campaign.db.get_results()

    # Get list containing data of the trace for the various param combination
    # and combination of params that generated it
    res_bucket = []
    for res_istance in res_data:
        res_id = res_istance['meta']['id']
        res_path = campaign.db.get_result_files(res_id)[trace_name]
        # Save both results and relative params
        new_df = pd.read_csv(filepath_or_buffer=res_path, header=0, delimiter='\t')
        # Improve data structure, keep just relevant data
        new_df = sanitize_dataframe(new_df, res_istance['params']['maxStart']*1e9) # sec to ns ns in the traces

        new_entry = {
            'results': new_df,
            'params': res_istance['params']
        }

        if len(new_df) != 0:
            res_bucket.append(new_entry)
        else:
            print('Empty trace found!')
            print('Path of the resource: '+ res_path)

    return res_bucket

def sanitize_dataframe(dataframe, treshold):
    # Remove trailing whitespaces from cols names
    dataframe = dataframe.rename(columns=lambda x: x.strip())
    # We want to keep trace just of packets transmitted after all apps started
    if 'tx_time' in dataframe.columns:
        dataframe = dataframe[dataframe['tx_time'] > treshold]   
    else:
        dataframe = dataframe[dataframe['time'] > treshold/1e9] # Need secs here

    # Go from IPI to source gen datarate
    # dataframe = dataframe[]

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


def band_allocation(trace_data):

    print('--Computing band allocation metric--')

    band_alloc = []

    for item in trace_data:
        # Find out total amount symbols available
        avail_sym = item['results']['frame'].iloc[-1] - item['results']['frame'].iloc[0]
        avail_sym = avail_sym*10*22 # Frames*subframes in a frame*symbols in a subframe
        # Get info regarding first CC
        item_cc0 = item['results'][item['results']['ccId'] == 0]
        used_sym_cc0 = item_cc0['symbol#'].sum()
        new_entry = {
            'cc0': used_sym_cc0/avail_sym,
            'params': item['params']
        }
        # If we are using CA, get also info regarding secondary CC
        if item['params']['mode'] == 2:
            item_cc1 = item['results'][item['results']['ccId'] == 1]
            used_sym_cc1 = item_cc1['symbol#'].sum()
            new_entry['cc1'] = used_sym_cc1/avail_sym

        # print('\n')
        # print(new_entry)
        # print('\n')

        band_alloc.append(new_entry)

    return band_alloc

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
        temp_path =  s_path + '/detailed/' + bearer_type + '_' + vs + '_' + str(vs_value)

        # Overlay mean throughput and appEnd 
        overlays = {
            'x': [temp_trace_data[0]['params']['appEnd']],
            'y': [temp_frame['value'].min()]
        }

        plot_line(packets_rx, 'Throughput [Mbit/s]', temp_title , temp_path, overlays)

    return out

def throughput_app(trace_data, bearer_type):
    """ 
    Computes the average throughput @ APP layer
    If parameters combination are provided, then only the simulations for
    such mean are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """
    print('--Computing per-user throughput--')

    ris = []
    for item in trace_data:
        g = (len(item['results'].index)*1024*8)/((item['params']['appEnd'] -
                                            item['params']['maxStart'])*1e6)  # computing overall throughput
        # computing per user throughput
        if bearer_type == 'urllc':
            single_g = g/(item['params']['numUrllcUes'])
        else:
            single_g = g/(item['params']['numEmbbUes'])

        ris.append({
            'mean': single_g,
            'params': item['params']
        })

    return ris


def delay_app(trace_data):
    """ 
    Computes the average delay @ APP layer.
    If parameters combination are provided, then only the simulations for
    such mean are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """
    print('--Computing average packet delay--')

    delay = []
    for item in trace_data:
        # get time of rx
        time_rx = item['results']['rx_time']
        # get time of tx
        time_tx = item['results']['tx_time']
        # packet delay
        pck_delay = (time_rx - time_tx)/1e6
        delay.append({
            # latency = mean of packet delay
            'mean': pck_delay.mean(),
            'var': pck_delay.std(),  # Output both latency and jitter
            'params': item['params']
        })

    return delay


def pkt_loss_app(trace_dl, trace_ul):
    """ 
    Computes the average delay @ APP layer.
    If parameters combination are provided, then only the simulations for
    such mean are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """
    print('--Computing average packet loss--')

    loss = []
    for index in range(len(trace_dl)):   # Amount of sim same for ul and dl
        sent = len(trace_ul[index]['results'].index)
        # Overall lost packets
        dropped = sent - len(trace_dl[index]['results'].index)
        # Percentage of packets lost
        dropped = dropped/len(trace_ul[index]['results'].index)
        loss.append({
            'mean': dropped,
            'params': trace_dl[index]['params']
        })

    return loss


# Small,  support functions

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
print('Both CA and non CA using f0=10GHz, f1=28Ghz')
'''
print('Metrics vs ccRatio')
ca_params = {'f0': 10e9, 'f1':28e9, 'mode': 2, 'embbUdpIPI': 82} # Rate eMBB = 100Mbit/s
no_ca_params = {'f0': 10e9, 'mode': 1, 'embbUdpIPI': 82} # Rate eMBB = 100Mbit/s

print('Computing URLLC stats')
plot_all_metrics(prot='urllc', param_ca=ca_params, param_no_ca=no_ca_params, versus='ccRatio')
print('Computing eMBB stats')
plot_all_metrics(prot='embb', param_ca=ca_params, param_no_ca=no_ca_params, versus='ccRatio')
'''

print('Metrics vs embbIPI')
print('Computing URLLC stats')
ca_params = {'f0': 10e9, 'f1':28e9, 'mode': 2}
no_ca_params = {'f0': 10e9, 'mode': 1}
plot_all_metrics(prot='urllc', param_ca=ca_params, param_no_ca=no_ca_params, versus='embbUdpIPI')
print('Computing eMBB stats')
plot_all_metrics(prot='embb', param_ca=ca_params, param_no_ca=no_ca_params, versus='embbUdpIPI')
print('--------')


print('CA using f0=10GHz, f1=28Ghz; non CA using f0=28Ghz')
'''
print('Metrics vs ccRatio')
ca_params = {'f0': 10e9, 'f1':28e9, 'mode': 2, 'embbUdpIPI': 82} # Rate eMBB = 100Mbit/s
no_ca_params = {'f0': 28e9, 'mode': 1, 'embbUdpIPI': 82} # Rate eMBB = 100Mbit/s

print('Computing URLLC stats')
plot_all_metrics(prot='urllc', param_ca=ca_params, param_no_ca=no_ca_params, versus='ccRatio')
print('Computing eMBB stats')
plot_all_metrics(prot='embb', param_ca=ca_params, param_no_ca=no_ca_params, versus='ccRatio')
'''

print('Metrics vs embbIPI')
print('Computing URLLC stats')
ca_params = {'f0': 10e9, 'f1':28e9, 'mode': 2}
no_ca_params = {'f0': 28e9, 'mode': 1}

plot_all_metrics(prot='urllc', param_ca=ca_params, param_no_ca=no_ca_params, versus='embbUdpIPI')
print('Computing eMBB stats')
plot_all_metrics(prot='embb', param_ca=ca_params, param_no_ca=no_ca_params, versus='embbUdpIPI')
print('--------')
