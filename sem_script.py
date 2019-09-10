import sem
ns_path = './'
ns_script = 'slicing'
ns_res_path = './slicing-res'

# Create the actual simulation campagins
campaign = sem.CampaignManager.new(
    ns_path, ns_script, ns_res_path, check_repo=False)

print(campaign)
# Obtain IPIs from rates
# eMMB
embb_packet_size = 1024
embb_rate_grid = range(80, 120, 10)
for rate in embb_rate_grid:
    rate = embb_packet_size*8/(rate*1e6)
embb_IPI_grid = list(embb_rate_grid)
# URLLC
urllc_packet_size = 1024
urllc_rate_grid = range(1, 1, 1)
for rate in urllc_rate_grid:
    rate = urllc_packet_size*8/(rate*1e6)
urllc_IPI_grid = list(urllc_rate_grid)

sim_duration = 2.0

params_grid = {
    'appEnd': sim_duration,
    'appStart': 0.3,
    'bsrTimer': 2.0,
    'bw': 5e8,
    'ccRatio': 0.25,
    'condition': 'a',
    'embbOn': True,
    'embbUdpIPI': embb_IPI_grid,
    'f0': 10e9,  # URLCC's CC
    'f1': 28e9,  # eMBB's CC
    'filePath': 'sim_res',
    'fileSize': 512000,
    'lambdaUrllc': 0.2,
    'mode': [1, 2],  # Test both without/with CA slicing
    'numEmbbUes': 10,
    'numEnbs': 1,
    'numUrllcUes': 10,
    'reorderingTimer': 1.0,
    'runSet': 1,
    'segmentSize': 536,
    'scenario': 'test-single-enb-moving',
    'scheduler': 1,  # Round Robin Scheduler
    'simTime': sim_duration,  # Low just for testing purposes, then at least 10
    'urllcOn': True,
    'urllcUdpIPI': 1,
    'useBuildings': False,  # Use MmWave3gppPropagationLossModel
    'useRlcAm': False,  # Use RLC UM
    'useUdp': True,
    'vMax': 1.0,
    'vMin': 10.0,
}

runs = 1
campaign.run_missing_simulations(sem.list_param_combinations(params_grid), runs)

# Get missing results for no CA and CC equal to 28GHz
params_grid.update(mode=1, f0=28e9)

campaign.run_missing_simulations(sem.list_param_combinations(params_grid), runs)