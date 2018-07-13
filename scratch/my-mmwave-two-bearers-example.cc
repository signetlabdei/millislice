#include <ns3/mmwave-helper.h>
#include <scratch/simulation-config/simulation-config.h>

using namespace ns3;

int
main (int argc, char *argv[])
{
	std::string filePath = "";
	double simTime = 0.5;

 // Create the component carriers
 Ptr<mmwave::MmWaveComponentCarrier> cc0 = SimulationConfig::CreateMmWaveCc (28e9, 0, 1, 1e9);
 Ptr<mmwave::MmWaveComponentCarrier> cc1 = SimulationConfig::CreateMmWaveCc (28e9, 1, 0, 1e9);

 NS_LOG_LOGIC ("Component Carrier " << (uint16_t)cc0->GetConfigurationParameters ()->GetCcId () <<
 					 		 " frequency : " << cc0->GetCenterFrequency ());

 NS_LOG_LOGIC ("Component Carrier " << (uint16_t)cc1->GetConfigurationParameters ()->GetCcId () <<
 					 		 " frequency : " << cc1->GetCenterFrequency ());

 // Create the ccMap
 std::map<uint8_t, mmwave::MmWaveComponentCarrier > ccMap;
 ccMap [0] = *cc0;
 ccMap [1] = *cc1;

 // Create and set the helper
 // First set UseCa = true, then NumberOfComponentCarriers
 Config::SetDefault("ns3::MmWaveHelper::UseCa",BooleanValue(true));
 Config::SetDefault("ns3::MmWaveHelper::NumberOfComponentCarriers",UintegerValue(2));
 Config::SetDefault("ns3::MmWaveHelper::EnbComponentCarrierManager",StringValue ("ns3::MmWaveBaRrComponentCarrierManager"));
 Config::SetDefault("ns3::MmWaveHelper::ChannelModel",StringValue("ns3::MmWave3gppChannel"));
 Config::SetDefault("ns3::MmWaveHelper::PathlossModel",StringValue("ns3::MmWave3gppPropagationLossModel"));
 Config::SetDefault("ns3::MmWaveHelper::RlcAmEnabled",BooleanValue(true));

 Ptr<mmwave::MmWaveHelper> helper = CreateObject<mmwave::MmWaveHelper> ();
 helper->SetCcPhyParams(ccMap);

 Ptr<mmwave::MmWavePointToPointEpcHelper> epcHelper = CreateObject<mmwave::MmWavePointToPointEpcHelper> ();
 helper->SetEpcHelper (epcHelper);

 // Create the eNB node
 NodeContainer enbNodes;
 enbNodes.Create(1);

 // Set eNB mobility
 SimulationConfig::SetConstantPositionMobility (enbNodes, Vector (0.0, 0.0, 30.0));

 // Install eNB device
 NetDeviceContainer enbNetDevices = helper->InstallEnbDevice (enbNodes);

 // Create UE node
 NodeContainer ueNodes;
 ueNodes.Create(1);

 // Set UE mobility
 SimulationConfig::SetConstantPositionMobility (ueNodes, Vector (10.0, 0.0, 5.0));

 // Install UE device
 NetDeviceContainer ueNetDevices = helper->InstallUeDevice(ueNodes);

 // Create the Internet
 std::pair<Ptr<Node>, Ipv4Address> remotePair = SimulationConfig::CreateInternet (epcHelper);
 Ptr<Node> remoteHost = remotePair.first;
 Ipv4Address remoteHostAddr = remotePair.second;

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
 SimulationConfig::SetupUdpApplication (remoteHost, ueIpIface.GetAddress (0), dlPort, interPacketInterval, 0.1, simTime);
 SimulationConfig::SetupUdpApplication (remoteHost, ueIpIface.GetAddress (0), dlPort1, interPacketInterval, 0.11, simTime);
 SimulationConfig::SetupUdpApplication (ueNodes.Get (0), remoteHostAddr, ulPort, interPacketInterval, 0.1, simTime);

 helper->EnableTraces();

 Simulator::Stop (Seconds (simTime));
 Simulator::Run ();
 Simulator::Destroy ();

 return 0;
}
