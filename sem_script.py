import sem
ns_path = "/WSL/ns3-mmwave-slicing"
ns_script = "slicing"    
ns_res_path = "/SEM"

# Create the actual simulation campagins
campaign = sem.CampaignManager.new(ns_path, ns_script, ns_res_path)
