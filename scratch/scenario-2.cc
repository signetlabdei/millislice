#include <ns3/mmwave-helper.h>
#include <scratch/simulation-config/simulation-config.h>

using namespace ns3;

int
main (int argc, char *argv[])
{
	std::string filePath = ""; // where to save the traces
	double simTime = 15; 		 // simulation time
	double centerFreq = 28e9;  // center frequency
	double bw = 1e9; 						 // total bandwidth
	double noCc = 1;  				 // number of CCs
	bool useRlcAm = true;			// choose RLC AM / UM
	double speed = 3.0; 		// UE speed
	double bsrTimer = 2.0;
	double reorderingTimer = 1.0;
	int runSet = 1;


	CommandLine cmd;
	cmd.AddValue ("centerFreq", "Central frequency", centerFreq);
	cmd.AddValue ("simTime", "Simulation time", simTime);
	cmd.AddValue ("filePath", "Where to put the output files", filePath);
	cmd.AddValue ("runSet", "Run number", runSet);
	cmd.AddValue ("noCc", "Number of CCs", noCc);
	cmd.AddValue ("bw", "Total bandwidth", bw);
	cmd.AddValue ("bsrTimer", "BSR timer [ms]", bsrTimer);
	cmd.AddValue ("reorderingTimer", "reordering timer [ms]", reorderingTimer);
	cmd.AddValue("useRlcAm", "Use rlc am", useRlcAm);
	cmd.Parse (argc, argv);

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
	std::string scenario = "UMa";
	std::string condition = "l"; // n = NLOS, l = LOS
	Config::SetDefault ("ns3::MmWave3gppPropagationLossModel::ChannelCondition", StringValue(condition));
	Config::SetDefault ("ns3::MmWave3gppPropagationLossModel::Scenario", StringValue(scenario));
	Config::SetDefault ("ns3::MmWave3gppPropagationLossModel::OptionalNlos", BooleanValue(false));
	Config::SetDefault ("ns3::MmWave3gppPropagationLossModel::Shadowing", BooleanValue(false)); // enable or disable the shadowing effect
	Config::SetDefault ("ns3::MmWave3gppPropagationLossModel::InCar", BooleanValue(false)); // enable or disable the shadowing effect

	Config::SetDefault ("ns3::MmWave3gppBuildingsPropagationLossModel::UpdateCondition", BooleanValue(true)); // enable or disable the LOS/NLOS update when the UE moves

	Config::SetDefault ("ns3::MmWave3gppChannel::UpdatePeriod", TimeValue (MilliSeconds (100))); // Set channel update period, 0 stands for no update.
	Config::SetDefault ("ns3::MmWave3gppChannel::CellScan", BooleanValue(false)); // Set true to use cell scanning method, false to use the default power method.
	Config::SetDefault ("ns3::MmWave3gppChannel::PortraitMode", BooleanValue(true)); // use blockage model with UT in portrait mode
	Config::SetDefault ("ns3::MmWave3gppChannel::NumNonselfBlocking", IntegerValue(4)); // number of non-self blocking obstacles
	Config::SetDefault ("ns3::MmWave3gppChannel::BlockerSpeed", DoubleValue(1)); // speed of non-self blocking obstacles

	// core network
	Config::SetDefault ("ns3::MmWavePointToPointEpcHelper::X2LinkDelay", TimeValue (MilliSeconds(1)));
	Config::SetDefault ("ns3::MmWavePointToPointEpcHelper::X2LinkDataRate", DataRateValue(DataRate ("1000Gb/s")));
	Config::SetDefault ("ns3::MmWavePointToPointEpcHelper::X2LinkMtu",  UintegerValue(10000));
	Config::SetDefault ("ns3::MmWavePointToPointEpcHelper::S1uLinkDelay", TimeValue (MicroSeconds(1000)));
	Config::SetDefault ("ns3::MmWavePointToPointEpcHelper::S1apLinkDelay", TimeValue (MilliSeconds(10))); // MME latency

 // Create the component carriers
 std::map<uint8_t, mmwave::MmWaveComponentCarrier> ccMap;
 for(int i = 0; i < noCc; i++)
 {
	  double ccFreq = centerFreq + bw/(2*noCc)*(2*i-noCc+1); // compute the CC frequency
	  Ptr<mmwave::MmWaveComponentCarrier> cc = SimulationConfig::CreateMmWaveCc (ccFreq,   // frequency
		 																																					 i, 		 	 // CC ID
																																							 i==0,	 	 // is primary?
																																						 	 bw/noCc); // bandwidth

		ccMap.insert(std::pair<uint8_t, mmwave::MmWaveComponentCarrier> (i, *cc));
 }

 // Create and set the helper
 // First set UseCa = true, then NumberOfComponentCarriers
 Config::SetDefault("ns3::MmWaveHelper::UseCa",BooleanValue(noCc>1));
 Config::SetDefault("ns3::MmWaveHelper::NumberOfComponentCarriers", UintegerValue(noCc));
 Config::SetDefault("ns3::MmWaveHelper::EnbComponentCarrierManager",StringValue ("ns3::MmWaveBaRrComponentCarrierManager"));
 Config::SetDefault("ns3::MmWaveHelper::ChannelModel",StringValue("ns3::MmWave3gppChannel"));
 Config::SetDefault("ns3::MmWaveHelper::PathlossModel",StringValue("ns3::MmWave3gppBuildingsPropagationLossModel"));
 Config::SetDefault("ns3::MmWaveHelper::RlcAmEnabled",BooleanValue(useRlcAm));

 Ptr<mmwave::MmWaveHelper> helper = CreateObject<mmwave::MmWaveHelper> ();
 helper->SetCcPhyParams(ccMap);

 Ptr<mmwave::MmWavePointToPointEpcHelper> epcHelper = CreateObject<mmwave::MmWavePointToPointEpcHelper> ();
 helper->SetEpcHelper (epcHelper);

 // Create the eNB node
 NodeContainer enbNodes;
 enbNodes.Create(1);

 // Create UE node
 NodeContainer ueNodes;
 ueNodes.Create(1);

 // Set eNB mobility
 SimulationConfig::SetConstantPositionMobility (enbNodes, Vector (50.0, 50.0, 10.0));

 // Set UE mobility
 Ptr<UniformRandomVariable> y = CreateObject<UniformRandomVariable> ();
 y->SetAttribute ("Min", DoubleValue (-5));
 y->SetAttribute ("Max", DoubleValue (0));

 SimulationConfig::SetConstantVelocityMobility (ueNodes, Vector (25.0, y->GetValue (), 1.6), Vector (0.0, speed, 0.0));

 // Create random buildings
 RandomBuildings::CreateRandomBuildings (0, 	// street width
	 																			 20, 	// block size
																				 100, // max x-axis
																				 50,	// max y-axis
																				 7);	// number of buildings

 // Install eNB device
 NetDeviceContainer enbNetDevices = helper->InstallEnbDevice (enbNodes);

 // Install UE device
 NetDeviceContainer ueNetDevices = helper->InstallUeDevice(ueNodes);

 // Create the Internet
 std::pair<Ptr<Node>, Ipv4Address> remotePair = SimulationConfig::CreateInternet (epcHelper);
 Ptr<Node> remoteHost = remotePair.first;
 //Ipv4Address remoteHostAddr = remotePair.second;

 // Install the Internet on the UE
 Ipv4InterfaceContainer ueIpIface = SimulationConfig::InstallUeInternet (epcHelper, ueNodes, ueNetDevices);

 helper->AttachToClosestEnb (ueNetDevices, enbNetDevices);

 // Create a dedicated bearer
 Ptr<NetDevice> ueDevice = ueNetDevices.Get(0);
 Ptr<mmwave::MmWaveUeNetDevice> ueMmWaveDevice = DynamicCast<mmwave::MmWaveUeNetDevice> (ueDevice);
 EpcTft::PacketFilter packetFilter; // Create a new tft packet filter
 packetFilter.localPortStart = 1235; // Set the filter policies
 packetFilter.localPortEnd = 1235;
 Ptr<EpcTft> tft = Create<EpcTft> (); // Create a new tft
 tft->Add (packetFilter); // Add the packet filter
 epcHelper->ActivateEpsBearer (ueDevice, ueMmWaveDevice->GetImsi (), tft, EpsBearer (EpsBearer::GBR_CONV_VOICE)); // Activate the bearer
 // All the packets that match the filter rule will be sent using this bearer.

 // Install and start applications on UEs and remote host
 AsciiTraceHelper asciiTraceHelper;
 Ptr<OutputStreamWrapper> dlStream = asciiTraceHelper.CreateFileStream (filePath + "PacketSinkDlRx.txt");
 Ptr<OutputStreamWrapper> ulStream = asciiTraceHelper.CreateFileStream (filePath + "PacketSinkUlRx.txt");

 uint16_t dlPort = 1234;
 uint16_t dlPort1 = 1235;
 uint16_t ulPort = 2000;
 SimulationConfig::SetupUdpPacketSink (ueNodes.Get (0), dlPort, 0.1, simTime, dlStream);
 SimulationConfig::SetupUdpPacketSink (ueNodes.Get (0), dlPort1, 0.1, simTime, dlStream);
 SimulationConfig::SetupUdpPacketSink (remoteHost, ulPort, 0.1, simTime, ulStream);

 uint16_t interPacketInterval = 1;
 SimulationConfig::SetupUdpApplication (remoteHost, ueIpIface.GetAddress (0), dlPort, interPacketInterval, 0.5, simTime);
 //SimulationConfig::SetupUdpApplication (remoteHost, ueIpIface.GetAddress (0), dlPort1, interPacketInterval, 0.11, simTime);
 //SimulationConfig::SetupUdpApplication (ueNodes.Get (0), remoteHostAddr, ulPort, interPacketInterval, 0.3, simTime);

 helper->EnableTraces();
 BuildingsHelper::MakeMobilityModelConsistent ();
 RandomBuildings::PrintGnuplottableBuildingListToFile (filePath + "buildings.txt");

 Simulator::Stop (Seconds (simTime));
 Simulator::Run ();
 Simulator::Destroy ();

 return 0;
}
