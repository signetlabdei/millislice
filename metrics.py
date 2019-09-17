import sem
import numpy as np
from textwrap import wrap

# Functions

def print_metric(metric_bucket):
    """
    Print metrics and params that generated them
    """
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

    return group_by_params(metric_bucket)

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
        run_bucket = []
        waste_bucket = []
        # Find sims with same params
        for sim in metric_bucket:
            if sim['params'] == temp_param:
                run_bucket.append(sim['values'])
                waste_bucket.append(sim)  
        # Create entry for same params, different runs	
        out_bucket.append({
            'values': run_bucket,
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
    simulations corresponding to such values.

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
    such values are taken into consideration for the computation.

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
            'values': single_g,
            'params': item['params']
        })

    return ris


def delay_app(bearer_type, param_comb=None):
    """ 
    Computes the average delay @ APP layer.
    If parameters combination are provided, then only the simulations for
    such values are taken into consideration for the computation.

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
            'values': [pck_delay.mean(), pck_delay.std()],  # Output both latency and jitter
            'params': item['params']
        })

    return delay


def pkt_loss_app(bearer_type, param_comb=None):
    """ 
    Computes the average delay @ APP layer.
    If parameters combination are provided, then only the simulations for
    such values are taken into consideration for the computation.

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
        dropped = len(trace_ul[index]['results']) - len(trace_dl[index]['results']) # Overall lost packets
        dropped = dropped/len(trace_ul[index]['results'])   # Percentage of packets lost
        loss.append({
            'values': dropped,
            'params': trace_dl[index]['params']
        })

    return loss

# Actual metrics computation
# Load the SEM campaign
campaign = sem.CampaignManager.load('./slicing-res')
print('--SEM campaign succesfully loaded--')
print('--Computing URLLC results--')
urllc_packet_loss = pkt_loss_app('embb')
urllc_packet_loss =  print_metric(urllc_packet_loss)
print(urllc_packet_loss)

