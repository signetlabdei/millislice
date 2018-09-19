#include <ns3/mmwave-helper.h>
#include <scratch/simulation-config/simulation-config.h>

using namespace ns3;

void SetupScenario (NodeContainer enbNodes, NodeContainer ueNodes, std::string scenario);

int
main (int argc, char *argv[])
{
	std::string filePath = ""; // where to save the traces
	int numEmbbUes = 1; // number of eMBB UEs
	int numUrllcUes = 1; // number of URLLC UEs
	double simTime = 15; 		 // simulation time
	double centerFreq = 28e9;  // center frequency
	double bw = 1e9; 						 // total bandwidth
	bool useRlcAm = true;			// choose RLC AM / UM
	//double speed = 3.0; 		// UE speed
	double bsrTimer = 2.0;
	double reorderingTimer = 1.0;
	int runSet = 1;
	int mode = 1; // mode 1 = 1 CC, no isolation
								// mode 2 = 2 CC, complete isolation
	double appStart = 0.3; // application start time
	double appEnd; // application start time
	bool urllcOn = true; // if true install the ftp application
	bool embbOn = true; // if true install the dash application
	bool useUdp = false; // if true use UDP client apps

	// Propagation loss model
	bool useBuildings = false; // if true use MmWave3gppBuildingsPropagationLossModel
	std::string condition = "a"; // MmWave3MmWave3gppPropagationLossModel condition, n = NLOS, l = LOS
	std::string scenario = "test"; // the simulation scenario

	// URLLC parameters
	double lambdaUrllc = 0.2; // average number of file/s
	int segmentSize = 536; // segment size in bytes
	int fileSize = 512000; // file size in bytes

	CommandLine cmd;
	cmd.AddValue ("centerFreq", "Central frequency", centerFreq);
	cmd.AddValue ("numEmbbUes", "Number of eMBB UEs", numEmbbUes);
	cmd.AddValue ("numUrllcUes", "Number of URLLC UEs", numUrllcUes);
	cmd.AddValue ("simTime", "Simulation time", simTime);
	cmd.AddValue ("filePath", "Where to put the output files", filePath);
	cmd.AddValue ("runSet", "Run number", runSet);
	cmd.AddValue ("bw", "Total bandwidth", bw);
	cmd.AddValue ("bsrTimer", "BSR timer [ms]", bsrTimer);
	cmd.AddValue ("reorderingTimer", "reordering timer [ms]", reorderingTimer);
	cmd.AddValue ("useRlcAm", "Use rlc am", useRlcAm);
	cmd.AddValue ("mode", "mode 1 = 1 CC, no isolation, mode 2 = 2 CC, complete isolation", mode);
	cmd.AddValue ("lambdaUrllc", "average number of file/s", lambdaUrllc);
	cmd.AddValue ("segmentSize", "segment size in bytes", segmentSize);
	cmd.AddValue ("fileSize", "file size in bytes", fileSize);
	cmd.AddValue ("appStart", "application start time", appStart);
	cmd.AddValue ("appEnd", "application end time", appEnd);
	cmd.AddValue ("urllcOn", "if true install the ftp application", urllcOn);
	cmd.AddValue ("embbOn", "if true install the dash application", embbOn);
	cmd.AddValue ("useBuildings", "if true use 3MmWave3gppBuildingsPropagationLossModel", useBuildings);
	cmd.AddValue ("condition", "MmWave3MmWave3gppPropagationLossModel condition, n = NLOS, l = LOS, a = all", condition);
	cmd.AddValue ("scenario", "the simulation scenario", scenario);
	cmd.AddValue ("useUdp", "if true use UDP client apps", useUdp);
	cmd.Parse (argc, argv);
	appEnd = simTime;

	// RNG
	RngSeedManager::SetSeed (1);
	RngSeedManager::SetRun (runSet);

	SimulationConfig::SetTracesPath (filePath);

	Config::SetDefault ("ns3::MmWaveFlexTtiMacScheduler::HarqEnabled", BooleanValue(true));
	Config::SetDefault ("ns3::MmWavePhyMacCommon::TbDecodeLatency", UintegerValue(200.0));
	Config::SetDefault ("ns3::MmWavePhyMacCommon::NumHarqProcess", UintegerValue(100));

	Config::SetDefault ("ns3::LteRlcAm::EnableAQM", BooleanValue(false));
	Config::SetDefault ("ns3::LteRlcAm::PollRetransmitTimer", TimeValue(MilliSeconds(2.0)));
	Config::SetDefault ("ns3::LteRlcAm::ReorderingTimer", TimeValue(MilliSeconds(reorderingTimer)));
	Config::SetDefault ("ns3::LteRlcAm::StatusProhibitTimer", TimeValue(MilliSeconds(1.0)));
	Config::SetDefault ("ns3::LteRlcAm::ReportBufferStatusTimer", TimeValue(MilliSeconds(bsrTimer)));
	Config::SetDefault ("ns3::LteRlcUm::ReorderingTimer", TimeValue(MilliSeconds(reorderingTimer)));
	Config::SetDefault ("ns3::LteRlcUm::ReportBufferStatusTimer", TimeValue(MilliSeconds(bsrTimer)));
	Config::SetDefault ("ns3::LteRlcUmLowLat::SendBsrWhenPacketTx", BooleanValue(true));

	//The available channel scenarios are 'RMa', 'UMa', 'UMi-StreetCanyon', 'InH-OfficeMixed', 'InH-OfficeOpen', 'InH-ShoppingMall'
	Config::SetDefault ("ns3::MmWave3gppPropagationLossModel::ChannelCondition", StringValue(condition));
	Config::SetDefault ("ns3::MmWave3gppPropagationLossModel::Scenario", StringValue("UMa"));
	Config::SetDefault ("ns3::MmWave3gppPropagationLossModel::OptionalNlos", BooleanValue(false));
	Config::SetDefault ("ns3::MmWave3gppPropagationLossModel::Shadowing", BooleanValue(false)); // enable or disable the shadowing effect
	Config::SetDefault ("ns3::MmWave3gppPropagationLossModel::InCar", BooleanValue(false)); // enable or disable the shadowing effect

	Config::SetDefault ("ns3::MmWave3gppBuildingsPropagationLossModel::UpdateCondition", BooleanValue(true)); // enable or disable the LOS/NLOS update when the UE moves

	Config::SetDefault ("ns3::MmWave3gppChannel::UpdatePeriod", TimeValue (MilliSeconds (100))); // Set channel update period, 0 stands for no update.
	Config::SetDefault ("ns3::MmWave3gppChannel::DirectBeam", BooleanValue(true)); // Set true to perform the beam in the exact direction of receiver node.
	Config::SetDefault ("ns3::MmWave3gppChannel::PortraitMode", BooleanValue(true)); // use blockage model with UT in portrait mode
	Config::SetDefault ("ns3::MmWave3gppChannel::NumNonselfBlocking", IntegerValue(4)); // number of non-self blocking obstacles
	Config::SetDefault ("ns3::MmWave3gppChannel::BlockerSpeed", DoubleValue(1)); // speed of non-self blocking obstacles

	// core network
	Config::SetDefault ("ns3::MmWavePointToPointEpcHelper::X2LinkDelay", TimeValue (MilliSeconds(1)));
	Config::SetDefault ("ns3::MmWavePointToPointEpcHelper::X2LinkDataRate", DataRateValue(DataRate ("1000Gb/s")));
	Config::SetDefault ("ns3::MmWavePointToPointEpcHelper::X2LinkMtu",  UintegerValue(10000));
	Config::SetDefault ("ns3::MmWavePointToPointEpcHelper::S1uLinkDelay", TimeValue (MicroSeconds(1000)));
	Config::SetDefault ("ns3::MmWavePointToPointEpcHelper::S1apLinkDelay", TimeValue (MilliSeconds(10))); // MME latency


	int numCc;
	bool splitDrb;

	switch(mode){
		case 1:
			numCc = 1;
			splitDrb = false;
			break;

		case 2:
			numCc = 2;
			splitDrb = true;
			break;

		default:
			NS_FATAL_ERROR ("Undefined mode");
	}

 // Create the component carriers
 std::map<uint8_t, mmwave::MmWaveComponentCarrier> ccMap;
 for(int i = 0; i < numCc; i++)
 {
	  double ccFreq = centerFreq + bw/(2*numCc)*(2*i-numCc+1); // compute the CC frequency
	  Ptr<mmwave::MmWaveComponentCarrier> cc = SimulationConfig::CreateMmWaveCc (ccFreq,   // frequency
		 																																					 i, 		 	 // CC ID
																																							 i==0,	 	 // is primary?
																																						 	 bw/numCc); // bandwidth

		ccMap.insert(std::pair<uint8_t, mmwave::MmWaveComponentCarrier> (i, *cc));
 }

 // Create and set the helper
 // First set UseCa = true, then NumberOfComponentCarriers
 Config::SetDefault("ns3::MmWaveHelper::UseCa",BooleanValue(numCc>1));
 Config::SetDefault("ns3::MmWaveHelper::NumberOfComponentCarriers", UintegerValue(numCc));
 if (splitDrb)
 {
	 Config::SetDefault("ns3::MmWaveHelper::EnbComponentCarrierManager",StringValue ("ns3::MmWaveSplitDrbComponentCarrierManager"));
 }
 else
 {
	 Config::SetDefault("ns3::MmWaveHelper::EnbComponentCarrierManager",StringValue ("ns3::MmWaveNoOpComponentCarrierManager"));
 }
 Config::SetDefault("ns3::MmWaveHelper::ChannelModel",StringValue("ns3::MmWave3gppChannel"));
 if (useBuildings)
 {
	 Config::SetDefault("ns3::MmWaveHelper::PathlossModel",StringValue("ns3::MmWave3gppBuildingsPropagationLossModel"));
 }
 else
 {
	 Config::SetDefault("ns3::MmWaveHelper::PathlossModel",StringValue("ns3::MmWave3gppPropagationLossModel"));
 }
 Config::SetDefault("ns3::MmWaveHelper::RlcAmEnabled",BooleanValue(useRlcAm));

 Ptr<mmwave::MmWaveHelper> helper = CreateObject<mmwave::MmWaveHelper> ();
 helper->SetCcPhyParams(ccMap);

 if (splitDrb)
 {
	 // Create the DRB - CC map
	 std::map<uint16_t, uint8_t> qciCcMap;
	 qciCcMap[9] = 0; // NGBR_VIDEO_TCP_DEFAULT
	 qciCcMap[81] = 1; // DCGBR_REMOTE_CONTROL
	 helper->SetQciCcMap (qciCcMap);
 }

 Ptr<mmwave::MmWavePointToPointEpcHelper> epcHelper = CreateObject<mmwave::MmWavePointToPointEpcHelper> ();
 helper->SetEpcHelper (epcHelper);

 // Create the eNB node
 NodeContainer enbNodes;
 enbNodes.Create (1);

 // Create UE node
 NodeContainer ueEmbbNodes;
 ueEmbbNodes.Create (numEmbbUes);

 NodeContainer ueUrllcNodes;
 ueUrllcNodes.Create (numUrllcUes);

 NodeContainer ueNodes;
 ueNodes.Add (ueEmbbNodes);
 ueNodes.Add (ueUrllcNodes);

 SetupScenario (enbNodes, ueNodes, "test");

 // Install eNB device
 NetDeviceContainer enbNetDevices = helper->InstallEnbDevice (enbNodes);

 // Install UE device
 NetDeviceContainer ueEmbbNetDevices = helper->InstallUeDevice(ueEmbbNodes);
 NetDeviceContainer ueUrllcNetDevices = helper->InstallUeDevice(ueUrllcNodes);

 NetDeviceContainer ueNetDevices;
 ueNetDevices.Add (ueEmbbNetDevices);
 ueNetDevices.Add (ueUrllcNetDevices);

 // Create the Internet
 std::pair<Ptr<Node>, Ipv4Address> remotePair = SimulationConfig::CreateInternet (epcHelper);
 Ptr<Node> remoteHost = remotePair.first;
 //Ipv4Address remoteHostAddr = remotePair.second;

 // Install the Internet on the UE
 Ipv4InterfaceContainer ueEmbbIpIface = SimulationConfig::InstallUeInternet (epcHelper, ueEmbbNodes, ueEmbbNetDevices);
 Ipv4InterfaceContainer ueUrllcIpIface = SimulationConfig::InstallUeInternet (epcHelper, ueUrllcNodes, ueUrllcNetDevices);

 Ipv4InterfaceContainer ueIpIface;
 ueIpIface.Add (ueEmbbIpIface);
 ueIpIface.Add (ueUrllcIpIface);

 helper->AttachToClosestEnb (ueNetDevices, enbNetDevices);

 uint16_t dlUrllcPort = 1235; // port for eMBB
 uint16_t dlEmbbPort = 1236; 	// port for URLLC

 // Create a dedicated bearer for the URLLC users
 for (uint8_t i = 0; i < ueUrllcNodes.GetN (); i++)
 {
 	Ptr<NetDevice> ueDevice = ueUrllcNetDevices.Get(0);
 	Ptr<mmwave::MmWaveUeNetDevice> ueMmWaveDevice = DynamicCast<mmwave::MmWaveUeNetDevice> (ueDevice);
 	EpcTft::PacketFilter packetFilter; // Create a new tft packet filter
 	packetFilter.localPortStart = dlUrllcPort; // Set the filter policies
 	packetFilter.localPortEnd = dlUrllcPort;
 	Ptr<EpcTft> tft = Create<EpcTft> (); // Create a new tft
 	tft->Add (packetFilter); // Add the packet filter
 	epcHelper->ActivateEpsBearer (ueDevice, ueMmWaveDevice->GetImsi (), tft, EpsBearer (EpsBearer::DCGBR_REMOTE_CONTROL)); // Activate the bearer
 	// All the packets that match the filter rule will be sent using this bearer.
 }

 // Create a dedicated bearer for the eMBB users
 for (uint8_t i = 0; i < ueEmbbNodes.GetN (); i++)
 {
	 Ptr<NetDevice> ueDevice = ueEmbbNetDevices.Get(0);
	 Ptr<mmwave::MmWaveUeNetDevice> ueMmWaveDevice = DynamicCast<mmwave::MmWaveUeNetDevice> (ueDevice);
	 EpcTft::PacketFilter packetFilter; // Create a new tft packet filter
	 packetFilter.localPortStart = dlEmbbPort; // Set the filter policies
	 packetFilter.localPortEnd = dlEmbbPort;
	 Ptr<EpcTft> tft = Create<EpcTft> (); // Create a new tft
	 tft->Add (packetFilter); // Add the packet filter
	 epcHelper->ActivateEpsBearer (ueDevice, ueMmWaveDevice->GetImsi (), tft, EpsBearer (EpsBearer::NGBR_VIDEO_TCP_DEFAULT)); // Activate the bearer
	 // All the packets that match the filter rule will be sent using this bearer.
 }


 // Install and start applications on UEs and remote host
 AsciiTraceHelper asciiTraceHelper;
 Ptr<OutputStreamWrapper> dlEmbbStream = asciiTraceHelper.CreateFileStream (filePath + "eMBB-app-trace.txt");
 Ptr<OutputStreamWrapper> dlUrllcStream = asciiTraceHelper.CreateFileStream (filePath + "urllc-app-trace.txt");

 // Install packet sink and application on eMBB nodes
 if (embbOn)
 {
	 for (uint8_t i = 0; i < ueEmbbNodes.GetN (); i++)
	 {
		 if (useUdp)
		 {
			 SimulationConfig::SetupUdpPacketSink (ueEmbbNodes.Get (i), // node
			 																			 dlEmbbPort, 					// port
																						 0.01, 						// start time
																						 simTime, 				// stop time
																						 dlEmbbStream); 			// trace file

			 SimulationConfig::SetupUdpApplication (remoteHost, 							// node
				 																			ueEmbbIpIface.GetAddress (i), // destination address
																							dlEmbbPort, 									// destination port
																							1000, 			// interpacket interval
																							0.3, 											// start time
																							simTime);									// stop time
		 }
		 else
		 {
			 SimulationConfig::SetupDashApplication (ueEmbbNodes.Get (i), // client node
	 																						remoteHost, 				 // server node
	 																						dlEmbbPort, 				 // port
	 																						i+1, 								 // video ID
	 																						appStart, 					 // start time
	 																						appEnd, 						 // stop time
	 																						dlEmbbStream);			 // trace file
		 }
	 }
 }

 // Install packet sink and application on URLLC nodes
 if (urllcOn)
 {
	for (uint8_t i = 0; i < ueUrllcNodes.GetN (); i++)
	 {
		 if (useUdp)
		 {
			 SimulationConfig::SetupUdpPacketSink (ueUrllcNodes.Get (i), // node
	 																					dlUrllcPort, 					// port
	 																					0.01, 						// start time
	 																					simTime, 				// stop time
	 																					dlUrllcStream); 			// trace file

	 		SimulationConfig::SetupUdpApplication (remoteHost, 							// node
	 																					 ueUrllcIpIface.GetAddress (i), // destination address
	 																					 dlUrllcPort, 									// destination port
	 																					 1000, 			// interpacket interval
	 																					 0.3, 											// start time
	 																					 simTime);									// stop time
		 }
		 else
		 {
			 SimulationConfig::SetupFtpModel3Application (remoteHost,                     //client node
		                                                ueUrllcNodes.Get (i),           // server node
		                                                ueUrllcIpIface.GetAddress (i),  // destination address
		                                                dlUrllcPort,                    // destination port
		                                        				lambdaUrllc,                    // lambda
		                                                fileSize,                       // file size
		                                                segmentSize,                    // segments size OBS: this is the size of the packets that the application forwards to the socket. This is not the size of the packets that are actually going to be transmitted.
		                                                appStart,                       // start time
		                                                appEnd,                        	// end time
		                                                dlUrllcStream);                 // trace file
		 }
	 }
 }


 helper->EnableTraces();
 BuildingsHelper::MakeMobilityModelConsistent ();
 PrintHelper::PrintGnuplottableBuildingListToFile (filePath + "buildings.txt");
 PrintHelper::PrintGnuplottableNodeListToFile (filePath + "devs.txt");

 Simulator::Stop (Seconds (simTime));
 Simulator::Run ();
 Simulator::Destroy ();

 return 0;
}

void
SetupScenario (NodeContainer enbNodes, NodeContainer ueNodes, std::string scenario)
{
	if (scenario == "road")
	{
		NS_LOG_INFO ("Setting up the road scenario");
		double speed = 3.0;

		// Set eNB mobility
		SimulationConfig::SetConstantPositionMobility (enbNodes, Vector (50.0, 50.0, 10.0));

		// Set UE mobility
		Ptr<UniformRandomVariable> y = CreateObject<UniformRandomVariable> ();
		y->SetAttribute ("Min", DoubleValue (-5));
		y->SetAttribute ("Max", DoubleValue (0));

		Ptr<UniformRandomVariable> x = CreateObject<UniformRandomVariable> ();
		x->SetAttribute ("Min", DoubleValue (25));
		x->SetAttribute ("Max", DoubleValue (75));

		for (uint8_t i = 0; i < ueNodes.GetN (); i++)
		{
		 SimulationConfig::SetConstantVelocityMobility (ueNodes.Get (i), Vector (x->GetValue (), y->GetValue (), 1.6), Vector (speed, 0.0, 0.0));
		}

		// Create random buildings
		RandomBuildings::CreateRandomBuildings (0, 	// street width
																					 20, 	// block size
																					 100, // max x-axis
																					 50,	// max y-axis
																					 7);	// number of buildings
	}
	else if (scenario == "test")
	{
		NS_LOG_INFO ("Setting up the test scenario");

		SimulationConfig::SetConstantPositionMobility (enbNodes, Vector (0.0, 0.0, 10.0));
		for (uint8_t i=0; i<ueNodes.GetN (); i++)
		{
			SimulationConfig::SetConstantPositionMobility (ueNodes.Get (i), Vector (100.0, 0.0, 1.5));
		}
	}
	else
	{
		NS_ABORT_MSG ("Undefined scenario");
	}

}
