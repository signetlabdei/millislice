import sem
import numpy as np
import os
import matplotlib
campaign = sem.CampaignManager.load('./slicing-res')
print(campaign)

#results = sem.CampaignManager.get_results_as_numpy_array()
# print(results)


def throughput(dict):
    for result in [campaign.db.get_complete_results({'nDevices': 4000})[0]]:
        print(result)


def delay_app():
    trace_str = 'test_eMBB-dl-app-trace.txt'
    trace_data = load_results(trace_name=trace_str)
    print(trace_data)


def load_results(trace_name, param=None):
    # Get the required files IDs
    if param is not None:
        res_data = campaign.db.get_results(param)
    else:
        res_data = campaign.db.get_results()

    # Get list containing data of the trace for the various param combination
    res_bucket = []
    for res_istance in res_data:
        res_id = res_istance['meta']['id']
        res_path = campaign.db.get_result_files(res_id)[trace_name]
        res_bucket.append(np.loadtxt(fname=res_path, skiprows=1)) # Skip structure spec row
    
    return res_bucket



# Test
delay_app()