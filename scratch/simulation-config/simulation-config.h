#include <ns3/core-module.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-helper.h>
#include <ns3/mobility-module.h>
#include <ns3/applications-module.h>
#include <ns3/buildings-module.h>
#include <ns3/dash-module.h>
#include <ns3/node-list.h>
#include <ns3/lte-module.h>
#include <ns3/mmwave-module.h>
#include <ns3/trace-source-accessor.h>

NS_LOG_COMPONENT_DEFINE("SimulationConfig");

namespace ns3
{

class SimulationConfig
{
public:
  static Ptr<mmwave::MmWaveComponentCarrier> CreateMmWaveCc(double freq, uint8_t ccId, bool isPrimary, double bw);
  static std::pair<Ptr<Node>, Ipv4Address> CreateInternet(Ptr<mmwave::MmWavePointToPointEpcHelper> epcHelper);
  static Ipv4InterfaceContainer InstallUeInternet(Ptr<mmwave::MmWavePointToPointEpcHelper> epcHelper, NodeContainer ueNodes, NetDeviceContainer ueNetDevices);
  static void SetConstantPositionMobility(NodeContainer nodes, Vector position);
  static void SetConstantVelocityMobility(Ptr<Node> node, Vector position, Vector velocity);
  static void SetRandomWalkMobility(Ptr<Node> node, Vector position, double vMin, double vMax, double rho);
  static void SetupUdpApplication(Ptr<Node> node, Ipv4Address address, uint16_t port, uint16_t interPacketInterval, Ptr<OutputStreamWrapper> stream, Ptr<RandomVariableStream> startTimeRv, double endTime);
  static void SetupFtpModel3Application(Ptr<Node> clientNode, Ptr<Node> serverNode, Ipv4Address address, uint16_t port, double lambda, uint32_t fileSize, uint32_t sendSize, double startTime, double endTime, Ptr<OutputStreamWrapper> stream);
  static void SetupUdpPacketSink(Ptr<Node> node, uint16_t port, double startTime, double endTime, Ptr<OutputStreamWrapper> stream);
  static void SetupDashApplication(Ptr<Node> senderNode, Ptr<Node> receiverNode, uint32_t port, uint8_t videoId, double startTime, double stopTime, Ptr<OutputStreamWrapper> stream);
  static void SetTracesPath(std::string filePath);

private:
  static void StartFileTransfer(Ptr<FileTransferApplication> ftpApp);
  static void EndFileTransfer(Ptr<ExponentialRandomVariable> readingTime, double endTime, Ptr<FileTransferApplication> ftpApp);
};

class CallbackSinks
{
public:
  static void RxSink(Ptr<OutputStreamWrapper> stream, Ptr<const Packet> packet, const Address &from = Address());
  static void TxSink(Ptr<OutputStreamWrapper> stream, Ptr<const Packet> packet, const Address &from = Address());
  // Specific UDP sinks, to enable increased telemetry and not break compatibility
  static void RxSinkUdp(Ptr<OutputStreamWrapper> stream, std::string context, Ptr<const Packet> packet, const Address &from = Address());
  static void TxSinkUdp(Ptr<OutputStreamWrapper> stream, Ptr<const Packet> packet, const Address &from = Address());
};

class RandomBuildings
{
public:
  static void CreateRandomBuildings(double streetWidth, double blockSize, double maxXAxis, double maxYAxis, uint32_t numBlocks);

private:
  static std::pair<Box, std::list<Box>> GenerateBuildingBounds(double xMin, double xMax, double yMin, double yMax, double maxBuildSize, std::list<Box> m_previousBlocks);
  static bool AreOverlapping(Box a, Box b);
  static bool OverlapWithAnyPrevious(Box box, std::list<Box> m_previousBlocks);
};

class PrintHelper
{
public:
  static void PrintGnuplottableBuildingListToFile(std::string filename);
  static void PrintGnuplottableNodeListToFile(std::string filename);
  static void UpdateGnuplottableNodeListToFile(std::string filename, Ptr<Node> node);
};

Ptr<mmwave::MmWaveComponentCarrier>
SimulationConfig::CreateMmWaveCc(double freq, uint8_t ccId, bool isPrimary, double bw)
{
  Ptr<mmwave::MmWavePhyMacCommon> phyMacConfig = CreateObject<mmwave::MmWavePhyMacCommon>();
  phyMacConfig->SetCentreFrequency(freq);
  phyMacConfig->SetCcId(ccId);
  phyMacConfig->SetNumChunkPerRB(phyMacConfig->GetNumChunkPerRb() * bw / 1e9);
  phyMacConfig->SetNumRefScPerSym(phyMacConfig->GetNumRefScPerSym() * bw / 1e9);

  Ptr<mmwave::MmWaveComponentCarrier> cc = CreateObject<mmwave::MmWaveComponentCarrier>();
  cc->SetConfigurationParameters(phyMacConfig);
  cc->SetAsPrimary(true);

  NS_LOG_INFO("-- Creating a CC --");
  NS_LOG_INFO("CC ID " << (uint16_t)ccId);
  NS_LOG_INFO("Frequency " << freq);
  NS_LOG_INFO("Is primary? " << isPrimary);
  NS_LOG_INFO("Bandwidth = " << bw);
  NS_LOG_INFO("NumChunkPerRb = " << phyMacConfig->GetNumChunkPerRb());
  NS_LOG_INFO("NumRefScPerSym = " << phyMacConfig->GetNumRefScPerSym() << "\n");

  return cc;
}

std::pair<Ptr<Node>, Ipv4Address>
SimulationConfig::CreateInternet(Ptr<mmwave::MmWavePointToPointEpcHelper> epcHelper)
{
  // Create the Internet by connecting remoteHost to pgw. Setup routing too
  Ptr<Node> pgw = epcHelper->GetPgwNode();

  // Create remotehost
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create(1);
  InternetStackHelper internet;
  internet.Install(remoteHostContainer);
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ipv4InterfaceContainer internetIpIfaces;

  Ptr<Node> remoteHost = remoteHostContainer.Get(0);
  // Create the Internet
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
  p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
  p2ph.SetChannelAttribute("Delay", TimeValue(MilliSeconds(0.01)));

  NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase("1.0.0.0", "255.255.0.0");
  internetIpIfaces = ipv4h.Assign(internetDevices);
  // interface 0 is localhost, 1 is the p2p device
  Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress(1);

  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
  remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.255.0.0"), 1);

  return std::pair<Ptr<Node>, Ipv4Address>(remoteHost, remoteHostAddr);
}

Ipv4InterfaceContainer
SimulationConfig::InstallUeInternet(Ptr<mmwave::MmWavePointToPointEpcHelper> epcHelper, NodeContainer ueNodes, NetDeviceContainer ueNetDevices)
{
  // Install the IP stack on the UEs
  InternetStackHelper internet;
  internet.Install(ueNodes);
  Ipv4InterfaceContainer ueIpIface;
  ueIpIface = epcHelper->AssignUeIpv4Address(ueNetDevices);
  // Assign IP address to UEs, and install applications
  // Set the default gateway for the UE
  Ipv4StaticRoutingHelper ipv4RoutingHelper;

  for (uint8_t i = 0; i < ueNodes.GetN(); i++)
  {
    Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting(ueNodes.Get(i)->GetObject<Ipv4>());
    ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
  }

  return ueIpIface;
}

void SimulationConfig::SetConstantPositionMobility(NodeContainer nodes, Vector position)
{
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
  positionAlloc->Add(position);
  MobilityHelper mobility;
  mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobility.SetPositionAllocator(positionAlloc);
  mobility.Install(nodes);
  BuildingsHelper::Install(nodes);
}

void SimulationConfig::SetConstantVelocityMobility(Ptr<Node> node, Vector position, Vector velocity)
{
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
  MobilityHelper mobility;
  positionAlloc->Add(position);
  mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
  mobility.SetPositionAllocator(positionAlloc);
  mobility.Install(node);
  node->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(velocity);
  BuildingsHelper::Install(node);
}

void SimulationConfig::SetRandomWalkMobility(Ptr<Node> node, Vector position, double vMin, double vMax, double rho)
{
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
  positionAlloc->Add(position);
  MobilityHelper mobility;
  // Not too elegant
  std::ostringstream paramUnifRv;
  paramUnifRv << "ns3::UniformRandomVariable[Min=" << vMin << "|Max=" << vMax << "]";
  mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                            "Mode", StringValue("Time"),
                            "Time", StringValue("4s"), // Time to wait before changing speed and/or direction of the walk
                            "Speed", StringValue(paramUnifRv.str()),
                            "Bounds", RectangleValue(Rectangle(-rho, rho, -rho, rho)));
  mobility.SetPositionAllocator(positionAlloc);
  mobility.Install(node);
  BuildingsHelper::Install(node);
}

void SimulationConfig::SetupUdpApplication(Ptr<Node> node, Ipv4Address address, uint16_t port, uint16_t interPacketInterval, Ptr<OutputStreamWrapper> stream, Ptr<RandomVariableStream> startTimeRv, double endTime)
{
  ApplicationContainer app;
  UdpClientHelper client(address, port);
  client.SetAttribute("Interval", TimeValue(MicroSeconds(interPacketInterval)));
  client.SetAttribute("MaxPackets", UintegerValue(10000000));

  app.Add(client.Install(node));
  // Sample starting time
  double startTime = startTimeRv->GetValue();
  app.Start(Seconds(startTime));
  app.Stop(Seconds(endTime));

  NS_LOG_INFO("Number of packets to send " << std::floor((endTime - startTime) / interPacketInterval * 1000));

  // Probably does not exist?
  app.Get(0)->TraceConnectWithoutContext("Tx", MakeBoundCallback(&CallbackSinks::TxSinkUdp, stream));
}

void SimulationConfig::SetupUdpPacketSink(Ptr<Node> node, uint16_t port, double startTime, double endTime, Ptr<OutputStreamWrapper> stream)
{
  ApplicationContainer app;
  PacketSinkHelper packetSinkHelper("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
  app.Add(packetSinkHelper.Install(node));
  app.Start(Seconds(startTime));
  app.Stop(Seconds(endTime));

  app.Get(0)->TraceConnect("Rx", std::to_string(node->GetId()), MakeBoundCallback(&CallbackSinks::RxSinkUdp, stream));
}

void SimulationConfig::SetupFtpModel3Application(Ptr<Node> clientNode, Ptr<Node> serverNode, Ipv4Address address, uint16_t port, double lambda, uint32_t fileSize, uint32_t sendSize, double startTime, double endTime, Ptr<OutputStreamWrapper> stream)
{
  // Install FTP application on client node
  ApplicationContainer clientApps;
  FileTransferHelper ftp("ns3::TcpSocketFactory", InetSocketAddress(address, port));
  ftp.SetAttribute("SendSize", UintegerValue(sendSize));
  ftp.SetAttribute("FileSize", UintegerValue(fileSize));
  clientApps.Add(ftp.Install(clientNode));
  clientApps.Start(Seconds(startTime));
  clientApps.Stop(Seconds(endTime - 0.1));

  // Install Packetink application on server node
  ApplicationContainer serverApps;
  PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
  serverApps = packetSinkHelper.Install(serverNode);
  serverApps.Start(Seconds(0));
  serverApps.Stop(Seconds(endTime));

  serverApps.Get(0)->TraceConnectWithoutContext("Rx", MakeBoundCallback(&CallbackSinks::RxSink, stream));
  clientApps.Get(0)->TraceConnectWithoutContext("Tx", MakeBoundCallback(&CallbackSinks::TxSink, stream));

  // Trigger data transfer
  Ptr<ExponentialRandomVariable> readingTime = CreateObject<ExponentialRandomVariable>();
  readingTime->SetAttribute("Mean", DoubleValue(1 / lambda));

  Ptr<FileTransferApplication> ftpApp = DynamicCast<FileTransferApplication>(clientApps.Get(0));
  ftpApp->SetFileTransferCompletedCallback(MakeBoundCallback(&SimulationConfig::EndFileTransfer, readingTime, endTime));

  double firstSend = readingTime->GetValue() + startTime;
  if (firstSend < endTime - 0.1)
  {
    Simulator::Schedule(Seconds(firstSend), &StartFileTransfer, ftpApp);
    NS_LOG_INFO("App " << ftpApp << " first file transmission scheduled at " << firstSend + Simulator::Now().GetSeconds());
  }
  else
  {
    NS_LOG_INFO("App " << ftpApp << " not enough time for any transmission (firstSend=" << firstSend << " endTime=" << endTime << ").");
  }
}

void SimulationConfig::EndFileTransfer(Ptr<ExponentialRandomVariable> readingTime, double endTime, Ptr<FileTransferApplication> ftpApp)
{
  double nextSend = readingTime->GetValue();
  if (nextSend + Simulator::Now().GetSeconds() < endTime)
  {
    Simulator::Schedule(Seconds(nextSend), &StartFileTransfer, ftpApp);
    NS_LOG_INFO("App " << ftpApp << " next file transmission scheduled at " << nextSend + Simulator::Now().GetSeconds());
  }
  else
  {
    NS_LOG_INFO("App " << ftpApp << " not enough time for further transmissions.");
  }
}

void SimulationConfig::StartFileTransfer(Ptr<FileTransferApplication> ftpApp)
{
  NS_LOG_INFO("App " << ftpApp << " start file transmission");
  ftpApp->SendFile();
}


void SimulationConfig::SetupDashApplication(Ptr<Node> clientNode, Ptr<Node> serverNode, uint32_t port, uint8_t videoId, double startTime, double stopTime, Ptr<OutputStreamWrapper> stream)
{
  Ptr<Ipv4> ipv4 = serverNode->GetObject<Ipv4>();
  Ipv4Address receiverIp = ipv4->GetAddress(1, 0).GetLocal();

  DashClientHelper client("ns3::TcpSocketFactory", InetSocketAddress(receiverIp, port), "ns3::DashClient");
  //client.SetAttribute ("MaxBytes", UintegerValue (maxBytes));
  client.SetAttribute("VideoId", UintegerValue(videoId));    // VideoId should be positive
  client.SetAttribute("TargetDt", TimeValue(Seconds(35.0))); // The target time difference between receiving and playing a frame.
  client.SetAttribute("window", TimeValue(Time("10s")));     // The window for measuring the average throughput (Time).
  ApplicationContainer clientApp = client.Install(clientNode);
  clientApp.Start(Seconds(startTime));
  clientApp.Stop(Seconds(stopTime - 0.1));

  DashServerHelper server("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
  ApplicationContainer serverApps = server.Install(serverNode);
  serverApps.Start(Seconds(0));
  serverApps.Stop(Seconds(stopTime));

  serverApps.Get(0)->TraceConnectWithoutContext("Tx", MakeBoundCallback(&CallbackSinks::TxSink, stream));
  clientApp.Get(0)->TraceConnectWithoutContext("Rx", MakeBoundCallback(&CallbackSinks::RxSink, stream));
}

void SimulationConfig::SetTracesPath(std::string filePath)
{
  Config::SetDefault("ns3::MmWaveBearerStatsCalculator::DlRlcOutputFilename", StringValue(filePath + "DlRlcStats.txt"));
  Config::SetDefault("ns3::MmWaveBearerStatsCalculator::UlRlcOutputFilename", StringValue(filePath + "UlRlcStats.txt"));
  Config::SetDefault("ns3::MmWaveBearerStatsCalculator::DlPdcpOutputFilename", StringValue(filePath + "DlPdcpStats.txt"));
  Config::SetDefault("ns3::MmWaveBearerStatsCalculator::UlPdcpOutputFilename", StringValue(filePath + "UlPdcpStats.txt"));
  Config::SetDefault("ns3::MmWavePhyRxTrace::OutputFilename", StringValue(filePath + "RxPacketTrace.txt"));
  Config::SetDefault("ns3::LteRlcAm::BufferSizeFilename", StringValue(filePath + "RlcAmBufferSize.txt"));
}

void RandomBuildings::CreateRandomBuildings(double streetWidth, double blockSize, double maxXAxis, double maxYAxis, uint32_t numBlocks)
{
  /* Create the building */
  double maxObstacleSize = blockSize - streetWidth;

  std::vector<Ptr<Building>> buildingVector;
  std::list<Box> m_previousBlocks;

  for (uint32_t buildingIndex = 0; buildingIndex < numBlocks; buildingIndex++)
  {
    Ptr<Building> building;
    building = Create<Building>();
    /* returns a vecotr where:
  		 * position [0]: coordinates for x min
  		 * position [1]: coordinates for x max
  		 * position [2]: coordinates for y min
  		 * position [3]: coordinates for y max
  		 */

    std::pair<Box, std::list<Box>> pairBuildings = RandomBuildings::GenerateBuildingBounds(0, maxXAxis - maxObstacleSize, 0, maxYAxis - maxObstacleSize, maxObstacleSize, m_previousBlocks);
    m_previousBlocks = std::get<1>(pairBuildings);
    Box box = std::get<0>(pairBuildings);

    Ptr<UniformRandomVariable> randomBuildingZ = CreateObject<UniformRandomVariable>();
    randomBuildingZ->SetAttribute("Min", DoubleValue(1.6));
    randomBuildingZ->SetAttribute("Max", DoubleValue(50));
    double buildingHeight = randomBuildingZ->GetValue();
    NS_LOG_INFO("Building height " << buildingHeight << "\n");

    building->SetBoundaries(Box(box.xMin, box.xMax,
                                box.yMin, box.yMax,
                                0.0, buildingHeight));

    building->SetNRoomsX(1);
    building->SetNRoomsY(1);
    building->SetNFloors(1);
    buildingVector.push_back(building);
  }
  /* END Create the building */
}
void CallbackSinks::RxSink(Ptr<OutputStreamWrapper> stream, Ptr<const Packet> packet, const Address &from)
{
  *stream->GetStream() << "Rx\t" << Simulator::Now().GetSeconds() << "\t" << packet->GetSize() << std::endl;
}

void CallbackSinks::TxSink(Ptr<OutputStreamWrapper> stream, Ptr<const Packet> packet, const Address &from)
{
  *stream->GetStream() << "Tx\t" << Simulator::Now().GetSeconds() << "\t" << packet->GetSize() << std::endl;
}

void CallbackSinks::RxSinkUdp(Ptr<OutputStreamWrapper> stream, std::string context, Ptr<const Packet> packet, const Address &from)
{
  // Get info about the packet
  Ptr<Packet> testPacket = packet->Copy(); // Need a non const reference to the packet
  SeqTsHeader seqTs;
  testPacket->RemoveHeader(seqTs);
  uint32_t currentSeqNmb = seqTs.GetSeq();
  Time currentTimestamp = seqTs.GetTs();
  int64_t nanosTimestamp = currentTimestamp.GetNanoSeconds();

  *stream->GetStream() << Simulator::Now().GetNanoSeconds() << "\t" << std::to_string(nanosTimestamp) << "\t" << packet->GetSize()
                       << "\t" << std::to_string(currentSeqNmb) << "\t" << context << std::endl;
}

void CallbackSinks::TxSinkUdp(Ptr<OutputStreamWrapper> stream, Ptr<const Packet> packet, const Address &to)
{
  // Get info about the packet
  Ptr<Packet> testPacket = packet->Copy(); // Need a non const reference to the packet
  SeqTsHeader seqTs;
  testPacket->RemoveHeader(seqTs);
  uint32_t currentSeqNmb = seqTs.GetSeq();
  // Get dest address info
  // Ipv4Address destIpv4 = Ipv4Address::ConvertFrom(to);

  *stream->GetStream() << Simulator::Now().GetNanoSeconds() << "\t" << packet->GetSize() << "\t" << std::to_string(currentSeqNmb) << std::endl;
}

std::pair<Box, std::list<Box>>
RandomBuildings::GenerateBuildingBounds(double xMin, double xMax, double yMin, double yMax, double maxBuildSize, std::list<Box> m_previousBlocks)
{

  Ptr<UniformRandomVariable> xMinBuilding = CreateObject<UniformRandomVariable>();
  xMinBuilding->SetAttribute("Min", DoubleValue(xMin));
  xMinBuilding->SetAttribute("Max", DoubleValue(xMax - 1)); // 1 m is the minimum size

  Ptr<UniformRandomVariable> yMinBuilding = CreateObject<UniformRandomVariable>();
  yMinBuilding->SetAttribute("Min", DoubleValue(yMin));
  yMinBuilding->SetAttribute("Max", DoubleValue(yMax - 1)); // 1 m is the minimum size

  Box box;
  uint32_t attempt = 0;
  do
  {
    NS_ASSERT_MSG(attempt < 100, "Too many failed attempts to position non-overlapping buildings. Maybe area too small or too many buildings?");
    box.xMin = xMinBuilding->GetValue();

    Ptr<UniformRandomVariable> xMaxBuilding = CreateObject<UniformRandomVariable>();
    xMaxBuilding->SetAttribute("Min", DoubleValue(box.xMin + 1)); // 1 m is the minimum size
    xMaxBuilding->SetAttribute("Max", DoubleValue(box.xMin + maxBuildSize));
    box.xMax = xMaxBuilding->GetValue();

    box.yMin = yMinBuilding->GetValue();

    Ptr<UniformRandomVariable> yMaxBuilding = CreateObject<UniformRandomVariable>();
    yMaxBuilding->SetAttribute("Min", DoubleValue(box.yMin + 1)); // 1 m is the minimum size
    yMaxBuilding->SetAttribute("Max", DoubleValue(box.yMin + maxBuildSize));
    box.yMax = yMaxBuilding->GetValue();

    ++attempt;
  } while (OverlapWithAnyPrevious(box, m_previousBlocks));

  NS_LOG_INFO("Building in coordinates (" << box.xMin << " , " << box.yMin << ") and (" << box.xMax << " , " << box.yMax << ") accepted after " << attempt << " attempts");
  m_previousBlocks.push_back(box);
  std::pair<Box, std::list<Box>> pairReturn = std::make_pair(box, m_previousBlocks);
  return pairReturn;
}

bool RandomBuildings::AreOverlapping(Box a, Box b)
{
  return !((a.xMin > b.xMax) || (b.xMin > a.xMax) || (a.yMin > b.yMax) || (b.yMin > a.yMax));
}

bool RandomBuildings::OverlapWithAnyPrevious(Box box, std::list<Box> m_previousBlocks)
{
  for (std::list<Box>::iterator it = m_previousBlocks.begin(); it != m_previousBlocks.end(); ++it)
  {
    if (AreOverlapping(*it, box))
    {
      return true;
    }
  }
  return false;
}

void PrintHelper::PrintGnuplottableBuildingListToFile(std::string filename)
{
  std::ofstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);
  if (!outFile.is_open())
  {
    NS_LOG_ERROR("Can't open file " << filename);
    return;
  }

  //outFile << "set xrange [0:100]" << std::endl;
  //outFile << "set yrange [0:100]" << std::endl;
  outFile << "unset key" << std::endl;
  outFile << "set grid" << std::endl;

  uint32_t index = 0;
  for (BuildingList::Iterator it = BuildingList::Begin(); it != BuildingList::End(); ++it)
  {
    ++index;
    Box box = (*it)->GetBoundaries();
    outFile << "set object " << index
            << " rect from " << box.xMin << "," << box.yMin
            << " to " << box.xMax << "," << box.yMax
            //<< " height " << box.zMin << "," << box.zMax
            << " front fs empty "
            << std::endl;
  }
}

void PrintHelper::PrintGnuplottableNodeListToFile(std::string filename)
{
  std::ofstream outFile;
  outFile.open(filename.c_str(), std::ios_base::out | std::ios_base::trunc);
  if (!outFile.is_open())
  {
    NS_LOG_ERROR("Can't open file " << filename);
    return;
  }
  for (NodeList::Iterator it = NodeList::Begin(); it != NodeList::End(); ++it)
  {
    Ptr<Node> node = *it;
    int nDevs = node->GetNDevices();
    for (int j = 0; j < nDevs; j++)
    {
      Ptr<LteUeNetDevice> uedev = node->GetDevice(j)->GetObject<LteUeNetDevice>();
      Ptr<mmwave::MmWaveUeNetDevice> mmuedev = node->GetDevice(j)->GetObject<mmwave::MmWaveUeNetDevice>();
      Ptr<mmwave::McUeNetDevice> mcuedev = node->GetDevice(j)->GetObject<mmwave::McUeNetDevice>();
      Ptr<LteEnbNetDevice> enbdev = node->GetDevice(j)->GetObject<LteEnbNetDevice>();
      Ptr<mmwave::MmWaveEnbNetDevice> mmenbdev = node->GetDevice(j)->GetObject<mmwave::MmWaveEnbNetDevice>();
      if (uedev)
      {
        Vector pos = node->GetObject<MobilityModel>()->GetPosition();
        outFile << "set label \"" << uedev->GetImsi()
                << "\" at " << pos.x << "," << pos.y << " left font \"Helvetica,8\" textcolor rgb \"black\" front point pt 1 ps 0.5 lc rgb \"black\" offset 0,0"
                << std::endl;

        Simulator::Schedule(Seconds(1), &PrintHelper::UpdateGnuplottableNodeListToFile, filename, node);
      }
      else if (mmuedev)
      {
        Vector pos = node->GetObject<MobilityModel>()->GetPosition();
        outFile << "set label \"" << mmuedev->GetImsi()
                << "\" at " << pos.x << "," << pos.y << " left font \"Helvetica,8\" textcolor rgb \"black\" front point pt 1 ps 0.5 lc rgb \"black\" offset 0,0"
                << std::endl;

        Simulator::Schedule(Seconds(1), &PrintHelper::UpdateGnuplottableNodeListToFile, filename, node);
      }
      else if (mcuedev)
      {
        Vector pos = node->GetObject<MobilityModel>()->GetPosition();
        outFile << "set label \"" << mcuedev->GetImsi()
                << "\" at " << pos.x << "," << pos.y << " left font \"Helvetica,8\" textcolor rgb \"black\" front point pt 1 ps 0.5 lc rgb \"black\" offset 0,0"
                << std::endl;

        Simulator::Schedule(Seconds(1), &PrintHelper::UpdateGnuplottableNodeListToFile, filename, node);
      }
      else if (enbdev)
      {
        Vector pos = node->GetObject<MobilityModel>()->GetPosition();
        outFile << "set label \"" << enbdev->GetCellId()
                << "\" at " << pos.x << "," << pos.y << " left font \"Helvetica,8\" textcolor rgb \"red\" front point pt 1 ps 0.5 lc rgb \"red\" offset 0,0"
                << std::endl;
      }
      else if (mmenbdev)
      {
        Vector pos = node->GetObject<MobilityModel>()->GetPosition();
        outFile << "set label \"" << mmenbdev->GetCellId()
                << "\" at " << pos.x << "," << pos.y << " left font \"Helvetica,8\" textcolor rgb \"red\" front point pt 1 ps 0.5 lc rgb \"red\" offset 0,0"
                << std::endl;
      }
    }
  }
}

void PrintHelper::UpdateGnuplottableNodeListToFile(std::string filename, Ptr<Node> node)
{
  std::ofstream outFile;
  outFile.open(filename.c_str(), std::ios_base::app);
  if (!outFile.is_open())
  {
    NS_LOG_ERROR("Can't open file " << filename);
    return;
  }
  Vector pos = node->GetObject<MobilityModel>()->GetPosition();
  outFile << "set label \""
          << "\" at " << pos.x << "," << pos.y << " left font \"Helvetica,8\" textcolor rgb \"black\" front point pt 1 ps 0.3 lc rgb \"black\" offset 0,0"
          << std::endl;

  Simulator::Schedule(Seconds(1), &PrintHelper::UpdateGnuplottableNodeListToFile, filename, node);
}
} // end namespace ns3
