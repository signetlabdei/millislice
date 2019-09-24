import sem
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import seaborn as sns
import pandas as pd


# Functions

def print_metric(metric_bucket, intro, just_mean=0):
    """
    Print metrics and params that generated them
    """
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
    
    sub_path = ''

    # Load results, specify params if given on input
    if param_no_ca is not None:
        trace_no_ca_dl = load_results(trace_name=trace_str_dl, param=param_no_ca)
        trace_no_ca_ul = load_results(trace_name=trace_str_ul, param=param_no_ca)
        # Subfoloder name
        sub_path += print_dict(param_no_ca)
        if param_ca is not None and param_ca is not param_no_ca:
            trace_ca_dl = load_results(trace_name=trace_str_dl, param=param_ca)
            trace_ca_ul = load_results(trace_name=trace_str_ul, param=param_ca)
            # Combine the traces lists:
            trace_dl = trace_no_ca_dl + trace_ca_dl
            trace_ul = trace_no_ca_ul + trace_ca_ul
            sub_path += 'versus_' + print_dict(param_ca)

        else:
            trace_dl = trace_no_ca_dl 
            trace_ul = trace_no_ca_ul            
    else:
        trace_dl = load_results(trace_name=trace_str_dl)
        trace_ul = load_results(trace_name=trace_str_ul)
        sub_path = 'no_spec_params'


    # Call lower level function
    plot_metric(metric_bucket=pkt_loss_app(trace_dl, trace_ul), metric='packet loss', prot=prot, s_path=sub_path, vs=versus)
    plot_metric(metric_bucket=throughput_app(trace_dl, bearer_type=prot), metric='throughput', s_path=sub_path, prot=prot, vs=versus)
    plot_metric(metric_bucket=delay_app(trace_dl),  metric='delay', prot=prot, s_path=sub_path, vs=versus)

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

def plot_metric(metric_bucket, metric, prot, s_path, vs=None):
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
        'versus': vs
    }

    metric_frame = pd.DataFrame(data=frame)
    metric_frame['mode'] =  metric_frame['mode'].replace(1, 'no CA')
    metric_frame['mode'] =  metric_frame['mode'].replace(2, 'CA')
    
    # Colors
    fig = plt.gcf()
    dark_palette = ['#465782', '#7a4e4f']
    light_palette = ['#90a5e0', '#c27a7c']
    sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})
    # Violin plot
    ax = sns.violinplot(data=metric_frame, y='metric', x='versus', hue='mode', palette=light_palette, split=True, inner='stick')
    #sns.stripplot(x="versus", y="metric", hue="mode", data=metric_frame, dodge=True, palette=dark_palette)

    # Overlay mean value
    # Obtain info regarding bounds
    left, right = plt.xlim()
    # Compute means
    means_bucket = compute_means(group_by_params(metric_bucket))
    if means_bucket[0]['params']['mode'] == 1:
        no_ca_mean = means_bucket[0]['mean']
        ca_mean = means_bucket[1]['mean']
    else:
        no_ca_mean = means_bucket[1]['mean']
        ca_mean = means_bucket[0]['mean']
    # Plot the values
    plt.plot([left, right], [no_ca_mean, no_ca_mean], color=dark_palette[0], linestyle='--', linewidth=1)
    plt.plot([left, right], [ca_mean, ca_mean], color=dark_palette[1], linestyle='--', linewidth=1)


    # Save, with the proper size
    if check_constant(versus_data):
        fig.set_size_inches(7, 7)
        # Set title and filename
        filename = f"{prot}_{metric}_CA_vs_nonCA.png"
        plot_title = f"{prot} {metric}"
    else:
        fig.set_size_inches(11, 7)
         # Set title and filename
        filename = f"{prot}_{metric}_vs{vs}_CA_vs_nonCA.png"
        plot_title = f"{prot} {metric} vs. {vs}"
        
    plt.ylabel(f"{metric} \n", fontsize=11)
    ax.set_xlabel('')
    plt.title(plot_title)

    # Save, create dir if doesn't exist       
    out_dir = f"./slicing-plots/{s_path}/"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir + filename)
   

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
        # If we have variance, save that as well
        var_bucket = []
        waste_bucket = []
        # Find sims with same params
        for sim in metric_bucket:
            if sim['params'] == temp_param:
                mean_bucket.append(sim['mean'])
                if('var' in metric_bucket[0]):
                    var_bucket.append(sim['var'])
                waste_bucket.append(sim)
        # Create entry for same params, different runs
        out_bucket.append({
            'mean': mean_bucket,
            'var': var_bucket,
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
        res_bucket.append({
            # Skip structure spec row
            'results': np.loadtxt(fname=res_path, skiprows=1),
            'params': res_istance['params']
        })

    return res_bucket


def throughput_app(trace_data, bearer_type, param_comb=None):
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
        g = (len(item['results'])*1024*8)/((item['params']['appEnd'] -
                                            item['params']['appStart'])*1e6)  # computing overall throughput
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


def delay_app(trace_data, param_comb=None):
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
        time_rx = item['results'][:, 0]
        # get time of tx
        time_tx = item['results'][:, 2]
        # packet delay
        pck_delay = (time_rx - time_tx)/1e6
        delay.append({
            # latency = mean of packet delay
            'mean': pck_delay.mean(),
            'var': pck_delay.std(),  # Output both latency and jitter
            'params': item['params']
        })

    return delay


def pkt_loss_app(trace_dl, trace_ul, param_comb=None):
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
        sent = len(trace_ul[index]['results'])
        # Overall lost packets
        dropped = sent - len(trace_dl[index]['results'])
        # Percentage of packets lost
        dropped = dropped/len(trace_ul[index]['results'])
        loss.append({
            'mean': dropped,
            'params': trace_dl[index]['params']
        })

    return loss


# Small,  support functions

def check_constant(bucket):
    return bucket[1:] == bucket[:-1]


def compute_means(metric_bucket):
    # Save original data
    out_bucket = copy.deepcopy(metric_bucket)

    for index in range(len(metric_bucket)):
        out_bucket[index]['mean'] = mean(metric_bucket[index]['mean'])
        if(len(metric_bucket[index]['var']) > 0):
            out_bucket[index]['var'] = mean(metric_bucket[index]['var'])
    return out_bucket



# Actual metrics computation
# Load the SEM campaign

"""
campaign = sem.CampaignManager.load('./slicing-res')
print('--SEM campaign succesfully loaded--')

print('--Computing URLLC results--')
urllc_packet_loss = pkt_loss_app('urllc')
urllc_packet_loss =  print_metric(urllc_packet_loss, 'URLLC PACKET LOSS \n', 1)
urllc_delay = delay_app('urllc')
urllc_delay =  print_metric(urllc_delay, 'URLLC DELAY \n', 1)
urllc_thr = throughput_app('urllc')
print_metric(urllc_thr, 'URLLC THROUGHPUT \n', 1)

print('--Computing EMBB results--')
embb_packet_loss = pkt_loss_app('embb')
embb_packet_loss =  print_metric(embb_packet_loss, 'EMBB PACKET LOSS \n', 1)
embb_delay = delay_app('embb')
embb_delay =  print_metric(embb_delay, 'EMBB DELAY \n', 1)
embb_thr = throughput_app('embb')
embb_thr =  print_metric(embb_thr, 'EMBB THROUGHPUT \n', 1)
"""
# Try plot
print('Both CA and non CA using f0=10GHz, f1=28Ghz')
print('Computing URLLC stats')
plot_all_metrics(prot='urllc', param_ca={'f0': 10e9, 'f1':28e9, 'mode': 2}, param_no_ca={'f0': 10e9, 'mode': 1}, versus='rho')
print('Computing eMBB stats')
plot_all_metrics(prot='embb', param_ca={'f0': 10e9, 'f1':28e9, 'mode': 2}, param_no_ca={'f0': 10e9, 'mode': 1}, versus='rho')

print('CA using f0=10GHz, f1=28Ghz; non CA using f0=28Ghz')
print('Computing URLLC stats')
plot_all_metrics(prot='urllc', param_ca={'f0': 10e9, 'f1':28e9, 'mode': 2}, param_no_ca={'f0': 28e9, 'mode': 1}, versus='rho')
print('Computing eMBB stats')
plot_all_metrics(prot='embb', param_ca={'f0': 10e9, 'f1':28e9, 'mode': 2}, param_no_ca={'f0': 28e9, 'mode': 1}, versus='rho')

