import sem
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from statistics import mean 
from operator import add, sub

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

def compute_means(metric_bucket):
    # Save original data
    out_bucket = copy.deepcopy(metric_bucket)

    for index in range(len(metric_bucket)):
        out_bucket[index]['mean'] = mean(metric_bucket[index]['mean'])
        if(len(metric_bucket[index]['var']) > 0):
            out_bucket[index]['var'] = mean(metric_bucket[index]['var']) 
    return out_bucket

def plot_metric(metric_bucket, versus, metric, shade):
    """ 
    Plots metric mean, CI and all run samples
    Args:
        versus (str): param to use on the x axis
    """
    # Obtain means
    metric_bucket = group_by_params(metric_bucket)
    delta_bucket = delta_ci_interval(metric_bucket)
    means_bucket = compute_means(metric_bucket)
    # Collect x's and y's
    x = []
    y = []
    for sim in means_bucket:
        x.append(sim['params'][versus])
        y.append(sim['mean'])

    # Plot means
    sns.set()
    sns.set_style("whitegrid")
    plt.plot(x, y, color=shade)
    # Plot all samples
    for x_val in x:
        y_buck = []
        for sim in metric_bucket:
            if sim['params'][versus] is x_val:
                y_buck.append(sim['mean'])
        plt.plot(x_val, y_buck, linestyle='', marker='.', markersize=5, color=shade)

    # Plot CIs
    # plt.fill_between(x, list(map(add, y, delta_bucket)), list(map(sub, y, delta_bucket)), color=shade, alpha=.4)
    print(y)
    print(x)
    # Get title
    plot_title =  f"{metric} vs. {versus}"
    plt.title(plot_title)
    plt.show()

def delta_ci_interval(metric_bucket):
    out_bucket = []
    for sim in metric_bucket:
        std_err = stats.sem(sim['mean'])
        delta = std_err * stats.t.ppf((1 + 0.95) / 2, len(sim['mean']) - 1)
        out_bucket.append(delta)

    return out_bucket

def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))

    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def check_constant(bucket):
    return bucket[1:] == bucket[:-1]

def group_by_params(metric_bucket):
    # Remove param specifying different runs
    for sim in metric_bucket:
        sim['params'].pop('RngRun', None) 
        sim['params'].pop('runSet', None) 
    out_bucket = []
    # Group sims having same param
    while(len(metric_bucket)> 0):
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


def throughput_app(bearer_type, param_comb=None):
    """ 
    Computes the average throughput @ APP layer
    If parameters combination are provided, then only the simulations for
    such mean are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """
    print('--Computing per-user throughput--')

    # Select proper trace file
    if bearer_type == 'urllc':
        trace_str = 'test_urllc-dl-app-trace.txt'
    else:
        trace_str = 'test_eMBB-dl-app-trace.txt'

    # Load results, specify params if given on input
    if param_comb is not None:
        trace_data = load_results(trace_name=trace_str, param=param_comb)
    else:
        trace_data = load_results(trace_name=trace_str)

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


def delay_app(bearer_type, param_comb=None):
    """ 
    Computes the average delay @ APP layer.
    If parameters combination are provided, then only the simulations for
    such mean are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """
    print('--Computing average packet delay--')

    # Select proper trace file
    if bearer_type == 'urllc':
        trace_str = 'test_urllc-dl-app-trace.txt'
    else:
        trace_str = 'test_eMBB-dl-app-trace.txt'

    # Load results, specify params if given on input
    if param_comb is not None:
        trace_data = load_results(trace_name=trace_str, param=param_comb)
    else:
        trace_data = load_results(trace_name=trace_str)

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


def pkt_loss_app(bearer_type, param_comb=None):
    """ 
    Computes the average delay @ APP layer.
    If parameters combination are provided, then only the simulations for
    such mean are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """
    print('--Computing average packet loss--')

    # Select proper trace file
    if bearer_type == 'urllc':
        trace_str_dl = 'test_urllc-dl-app-trace.txt'
        trace_str_ul = 'test_urllc-ul-sink-app-trace.txt'
    else:
        trace_str_dl = 'test_eMBB-dl-app-trace.txt'
        trace_str_ul = 'test_eMBB-ul-app-trace.txt'

    # Load results, specify params if given on input
    if param_comb is not None:
        trace_dl = load_results(trace_name=trace_str_dl, param=param_comb)
        trace_ul = load_results(trace_name=trace_str_ul, param=param_comb)
    else:
        trace_dl = load_results(trace_name=trace_str_dl)
        trace_ul = load_results(trace_name=trace_str_ul)

    loss = []
    for index in range(len(trace_dl)):   # Amount of sim same for ul and dl
        sent = len(trace_ul[index]['results'])
        dropped = sent - len(trace_dl[index]['results']) # Overall lost packets
        dropped = dropped/len(trace_ul[index]['results'])   # Percentage of packets lost
        loss.append({
            'mean': dropped,
            'params': trace_dl[index]['params']
        })

    return loss

# Actual metrics computation
# Load the SEM campaign
campaign = sem.CampaignManager.load('./slicing-res')
print('--SEM campaign succesfully loaded--')
"""
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
plot_metric(throughput_app('urllc'), 'mode', 'Throughput', [0, 0, 0])
plot_metric(delay_app('urllc'), 'mode', 'Delay', [0, 0, 0])
plot_metric(pkt_loss_app('urllc'), 'mode', 'Packet loss', [0, 0, 0])

plot_metric(throughput_app('embb'), 'mode', 'Throughput', [0, 0, 0])
plot_metric(delay_app('embb'), 'mode', 'Delay', [0, 0, 0])
plot_metric(pkt_loss_app('embb'), 'mode', 'Packet loss', [0, 0, 0])