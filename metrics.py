import sem
import numpy as np
from textwrap import wrap

# Functions

def print_metric(metric_bucket, intro, output_str=None):
    """
    Print metrics and params that generated them
    """
    # Place similar params close together
    metric_bucket.sort(key=ret_rng_run)
    if output_str is None:
        output_str = ''
    # Find out which param is changing
    params_mask = []
    params_list = list(metric_bucket[0]['params'].keys())   # list of params
    for item in params_list:
        # Check if that param is the same for all simulations
        temp_bucket = []
        for sim in metric_bucket:
            temp_bucket.append(sim['params'][item])
        params_mask.append(check_constant(temp_bucket))

    output_str += 'Constant simulations params: \n'
    const_par = np.array(params_list)[np.logical_not(params_mask)]
    for param in const_par:
        output_str += (param + ': ')
        output_str += (str(metric_bucket[0]['params'][param]) + '\t')
    output_str += '\n'

    output_str += 'Metric values: \n' 
    var_par = np.array(params_list)[params_mask]
    for sim in metric_bucket:
        output_str += (intro + str(sim['values']) + '\n')
        for param in var_par:
            output_str += (param + ': ')
            output_str += (str(sim['params'][param]) + '\t')
        output_str += '\n -------------- \n'

    return output_str

def check_constant(bucket):
    return bucket[1:] != bucket[:-1]

def ret_rng_run(elem):
    return elem['params']['RngRun']

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
        g = (len(item['results'])*1024)/((item['params']['appEnd'] -
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

out_str = ''
print('--Computing URLLC results--')
urllc_packet_loss = pkt_loss_app('urllc')
out_str = print_metric(urllc_packet_loss, 'URLLC packet loss: \n ', out_str)
urllc_delay = delay_app('urllc')
out_str = print_metric(urllc_delay, 'URLLC latency, jitter: \n', out_str)
urllc_throughput = throughput_app('urllc')
out_str = print_metric(urllc_throughput, 'URLLC throughput: \n', out_str)

print('--Computing eMBB results--')
embb_packet_loss = pkt_loss_app('embb')
out_str = print_metric(embb_packet_loss, 'eMBB packet loss: ', out_str)
embb_delay = delay_app('embb')
out_str = print_metric(embb_delay, 'eMBB latency, jitter: ', out_str)
embb_throughput = throughput_app('embb')
out_str = print_metric(embb_throughput, 'eMBB throughput: ', out_str)
print(out_str)

# Output to file
out_file = open("slicing-res/metrics_output.txt","a") 
out_file.write(out_str)