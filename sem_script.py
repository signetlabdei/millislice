import sem
ns_path = './'
ns_script = 'slicing'
ns_res_path = './slicing-res'

# Create the actual simulation campagins
campaign = sem.CampaignManager.new(
    ns_path, ns_script, ns_res_path, check_repo=False)

print(campaign)

params_grid = {
    'bw': 5e8,
    'f0': 10e9,  # URLCC's CC
    'f1': 28e9,  # eMBB's CC
    'mode': [1, 2], # Test both without/with CA slicing 
    'numEmbbUes': 10,
    'numUrllcUes': 10,
    'scenario': 'test-single-enb-moving',
    'scheduler': 1,  # Round Robin Scheduler
    'simTime': 2.0,  # Low just for testing purposes, then at least 10
    'useBuildings': False,  # Use MmWave3gppPropagationLossModel
    'useRlcAm': False,  # Use RLC UM
    'úseUdp': True,
    'vMax': 1.0,
    'vMin': 10.0,
}
