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
    results = campaign.db.get_complete_results()
    # Get the required output traces
    trace_bucket = []
    for result in results:
        dl_app_trace = result['output']['test_eMBB-dl-app-trace.txt']
        trace_bucket.append(dl_app_trace)
        print(dl_app_trace)
        print('---------------------')

delay_app()
