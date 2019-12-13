import sem
ns_path = './'
ns_script = 'slicing'
ns_res_path = './slicing-res'

# Create the actual simulation campagins
campaign = sem.CampaignManager.new(
    ns_path, ns_script, ns_res_path, check_repo=False, optimized=True, runner_type='LptRunner')

# Obtain IPIs from rates
# eMMB
embb_packet_size = 1024
embb_rate_grid = list(range(100, 101, 1))
embb_IPI_grid = []
for rate in embb_rate_grid:
    # Mbit/s to IPI in microseconds
    temp_IPI = embb_packet_size*8/(rate)
    embb_IPI_grid.append(int(round(temp_IPI)))
# URLLC
urllc_packet_size = 1024
urllc_rate_grid = list(range(1, 2, 1))
urllc_IPI_grid = []
for rate in urllc_rate_grid:
    temp_IPI = urllc_packet_size*8/(rate)
    urllc_IPI_grid.append(int(round(temp_IPI)))


# Set amount of simulation time
sim_duration = 7
runs = 25

params_grid = {
    'appEnd': sim_duration - 2,
    'minStart': 0.3,
    'maxStart': 0.4,
    'bsrTimer': 1.0,
    'bw': 5e8,
    'ccRatio': 0.5,
    'condition': 'a',
    'ccMan': 0,
    'embbOn': True,
    'embbUdpIPI': embb_IPI_grid,
    'f0': 28e9,  # eMBB's CC
    'f1': 10e9,  # URLLC's CC
    'filePath': 'test_',
    'fileSize': 512000,
    'lambdaUrllc': 0.2,
    'mode': [1, 2],  # Test both without/with CA slicing
    'numEmbbUes': 10,
    'numEnbs': 1,
    'numUrllcUes': 10,
    'rho': 200,
    'reorderingTimer': 1.0,
    'runSet': list(range(runs)),
    'RngRun' : 1,
    'segmentSize': 536,
    'scenario': 'test-single-enb-moving',
    'scheduler': 1,  # Round Robin Scheduler
    'simTime': sim_duration,  # Low just for testing purposes, then at least 10
    'urllcOn': True,
    'urllcUdpIPI': urllc_IPI_grid,
    'useBuildings': False,  # Use MmWave3gppPropagationLossModel
    'useRlcAm': True,  # Use RLC UM
    'useUdp': True,
    'vMax': 10.0,
    'vMin': 1.0,
}

print(params_grid)
campaign.run_missing_simulations(sem.list_param_combinations(params_grid))

# Get missing results for CA, Slciing CC Manager
params_grid.update(mode=2, ccMan=1)
campaign.run_missing_simulations(sem.list_param_combinations(params_grid))

