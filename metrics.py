import sem
import numpy as np
import matplotlib

# Functions


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
    print('--Computing throughput--')
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

    # Placeholder
    print(trace_data)


def delay_app(bearer_type, param_comb=None):
    """ 
    Computes the average delay @ APP layer.
    If parameters combination are provided, then only the simulations for
    such values are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """
    print('--Computing packet delay--')
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

    trace_data = load_results(trace_name=trace_str)

    # Placeholder
    print(trace_data)

# Actual metrics computation


# Load the SEM campaign
campaign = sem.CampaignManager.load('./slicing-res')
print('--SEM campaign succesfully loaded--')
test_no_param = delay_app('emmb')
