#include <ns3/core-module.h>
#include <ns3/mmwave-component-carrier.h>
#include <ns3/mmwave-point-to-point-epc-helper.h>
#include <ns3/internet-module.h>
#include <ns3/point-to-point-helper.h>
#include <ns3/mobility-module.h>
#include <ns3/applications-module.h>
#include <ns3/buildings-module.h>
#include <ns3/rmcat-sender.h>
#include <ns3/rmcat-receiver.h>
#include <ns3/dash-module.h>

NS_LOG_COMPONENT_DEFINE ("SimulationConfig");

namespace ns3{

  class SimulationConfig
  {
    public:
      static Ptr<mmwave::MmWaveComponentCarrier> CreateMmWaveCc (double freq, uint8_t ccId, bool isPrimary, double bw);
      static std::pair<Ptr<Node>, Ipv4Address> CreateInternet (Ptr<mmwave::MmWavePointToPointEpcHelper> epcHelper);
      static Ipv4InterfaceContainer InstallUeInternet (Ptr<mmwave::MmWavePointToPointEpcHelper> epcHelper, NodeContainer ueNodes, NetDeviceContainer ueNetDevices);
      static void SetConstantPositionMobility (NodeContainer nodes, Vector position);
      static void SetConstantVelocityMobility (Ptr<Node> node, Vector position, Vector velocity);
      static void SetupUdpApplication (Ptr<Node> node, Ipv4Address address, uint16_t port, uint16_t interPacketInterval, double startTime, double endTime);
      static void SetupFtpModel3Application (Ptr<Node> clientNode, Ptr<Node> serverNode, Ipv4Address address, uint16_t port, double lambda, uint32_t fileSize, uint32_t sendSize, double startTime, double endTime, Ptr<OutputStreamWrapper> stream);
      static void SetupUdpPacketSink (Ptr<Node> node, uint16_t port, double startTime, double endTime, Ptr<OutputStreamWrapper> stream);
      static void InstallRmcatApps (bool nada, Ptr<Node> sender, Ptr<Node> receiver, uint16_t port, float initBw, float minBw, float maxBw, float startTime, float stopTime);
      static void SetupDashApplication (Ptr<Node> senderNode, Ptr<Node> receiverNode, uint32_t port, uint8_t videoId,  double startTime, double stopTime, Ptr<OutputStreamWrapper> stream);
      static void SetTracesPath (std::string filePath);

    private:
      static void StartFileTransfer (ApplicationContainer clientApps, Ptr<ExponentialRandomVariable> ftpArrivals, double endTime);
  };

  class CallbackSinks
  {
    public:
      static void RxSink (Ptr<OutputStreamWrapper> stream, Ptr<const Packet> packet, const Address &from);
  };

  class RandomBuildings
  {
    public:
      static void CreateRandomBuildings (double streetWidth, double blockSize, double maxXAxis, double maxYAxis, uint32_t numBlocks);
      static void PrintGnuplottableBuildingListToFile (std::string filename);

    private:
      static std::pair<Box, std::list<Box>> GenerateBuildingBounds(double xMin, double xMax, double yMin, double yMax, double maxBuildSize, std::list<Box> m_previousBlocks );
      static bool AreOverlapping(Box a, Box b);
      static bool OverlapWithAnyPrevious(Box box, std::list<Box> m_previousBlocks);

  };

  Ptr<mmwave::MmWaveComponentCarrier>
  SimulationConfig::CreateMmWaveCc (double freq, uint8_t ccId, bool isPrimary, double bw)
  {
    Ptr<mmwave::MmWavePhyMacCommon> phyMacConfig = CreateObject<mmwave::MmWavePhyMacCommon> ();
    phyMacConfig->SetCentreFrequency (freq);
    phyMacConfig->SetCcId (ccId);
    phyMacConfig->SetNumChunkPerRB (phyMacConfig->GetNumChunkPerRb () * bw/1e9);
    phyMacConfig->SetNumRefScPerSym (phyMacConfig->GetNumRefScPerSym () *bw/1e9);

    Ptr<mmwave::MmWaveComponentCarrier> cc = CreateObject<mmwave::MmWaveComponentCarrier> ();
    cc->SetConfigurationParameters(phyMacConfig);
    cc->SetAsPrimary(true);

    NS_LOG_INFO ("-- Creating a CC --");
    NS_LOG_INFO ("CC ID " << (uint16_t)ccId);
    NS_LOG_INFO ("Frequency " << freq);
    NS_LOG_INFO ("Is primary? " << isPrimary);
    NS_LOG_INFO ("Bandwidth = " << bw);
    NS_LOG_INFO ("NumChunkPerRb = " << phyMacConfig->GetNumChunkPerRb ());
    NS_LOG_INFO ("NumRefScPerSym = " << phyMacConfig->GetNumRefScPerSym () << "\n");

    return cc;
  }

  std::pair<Ptr<Node>, Ipv4Address>
  SimulationConfig::CreateInternet (Ptr<mmwave::MmWavePointToPointEpcHelper> epcHelper)
  {
    // Create the Internet by connecting remoteHost to pgw. Setup routing too
    Ptr<Node> pgw = epcHelper->GetPgwNode ();

    // Create remotehost
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create (1);
    InternetStackHelper internet;
    internet.Install (remoteHostContainer);
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ipv4InterfaceContainer internetIpIfaces;

    Ptr<Node> remoteHost = remoteHostContainer.Get (0);
    // Create the Internet
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("100Gb/s")));
    p2ph.SetDeviceAttribute ("Mtu", UintegerValue (1500));
    p2ph.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (0.01)));

    NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase ("1.0.0.0", "255.255.0.0");
    internetIpIfaces = ipv4h.Assign (internetDevices);
    // interface 0 is localhost, 1 is the p2p device
    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress (1);

    Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
    remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.255.0.0"), 1);

    return std::pair<Ptr<Node>, Ipv4Address> (remoteHost, remoteHostAddr);
  }

  Ipv4InterfaceContainer
  SimulationConfig::InstallUeInternet (Ptr<mmwave::MmWavePointToPointEpcHelper> epcHelper, NodeContainer ueNodes, NetDeviceContainer ueNetDevices)
  {
    // Install the IP stack on the UEs
    InternetStackHelper internet;
    internet.Install (ueNodes);
    Ipv4InterfaceContainer ueIpIface;
    ueIpIface = epcHelper->AssignUeIpv4Address (ueNetDevices);
    // Assign IP address to UEs, and install applications
    // Set the default gateway for the UE
    Ipv4StaticRoutingHelper ipv4RoutingHelper;

    for (uint8_t i = 0; i < ueNodes.GetN (); i++)
    {
      Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNodes.Get (i)->GetObject<Ipv4> ());
      ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }

    return ueIpIface;
  }

  void
  SimulationConfig::SetConstantPositionMobility (NodeContainer nodes, Vector position)
  {
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
    positionAlloc->Add (position);
    MobilityHelper mobility;
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.SetPositionAllocator(positionAlloc);
    mobility.Install (nodes);
    BuildingsHelper::Install (nodes);
  }

  void
  SimulationConfig::SetConstantVelocityMobility (Ptr<Node> node, Vector position, Vector velocity)
  {
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
    MobilityHelper mobility;
    positionAlloc->Add (position);
    mobility.SetMobilityModel ("ns3::ConstantVelocityMobilityModel");
    mobility.SetPositionAllocator(positionAlloc);
    mobility.Install (node);
    node->GetObject<ConstantVelocityMobilityModel> ()->SetVelocity (velocity);
    BuildingsHelper::Install (node);
  }


  void
  SimulationConfig::SetupUdpApplication (Ptr<Node> node, Ipv4Address address, uint16_t port, uint16_t interPacketInterval, double startTime, double endTime)
  {
    ApplicationContainer app;
    UdpClientHelper client (address, port);
    client.SetAttribute ("Interval", TimeValue (MicroSeconds(interPacketInterval)));
    client.SetAttribute ("MaxPackets", UintegerValue(10000000));

    app.Add (client.Install (node));
    app.Start (Seconds (startTime));
    app.Stop (Seconds (endTime));

    NS_LOG_INFO ("Number of packets to send " << std::floor((endTime-startTime)/interPacketInterval*1000));
  }

  void
  SimulationConfig::SetupUdpPacketSink (Ptr<Node> node, uint16_t port, double startTime, double endTime, Ptr<OutputStreamWrapper> stream)
  {
    ApplicationContainer app;
    PacketSinkHelper packetSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), port));
    app.Add (packetSinkHelper.Install (node));
    app.Start (Seconds (startTime));
    app.Stop (Seconds (endTime));

    app.Get(0)->TraceConnectWithoutContext("Rx", MakeBoundCallback (&CallbackSinks::RxSink, stream));
  }

  void
  SimulationConfig::SetupFtpModel3Application (Ptr<Node> clientNode, Ptr<Node> serverNode, Ipv4Address address, uint16_t port, double lambda, uint32_t fileSize, uint32_t sendSize, double startTime, double endTime, Ptr<OutputStreamWrapper> stream)
  {
    // Install FTP application on client node
    ApplicationContainer clientApps;
    FileTransferHelper ftp ("ns3::TcpSocketFactory", InetSocketAddress (address, port));
    ftp.SetAttribute ("SendSize", UintegerValue (sendSize));
    ftp.SetAttribute ("FileSize", UintegerValue (fileSize));
    clientApps.Add (ftp.Install (clientNode));
    clientApps.Start (Seconds (startTime));
    clientApps.Stop (Seconds (endTime));

    // Install Packetink application on server node
    ApplicationContainer serverApps;
    PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), port));
    serverApps = packetSinkHelper.Install (serverNode);
    serverApps.Start (Seconds (startTime));
    serverApps.Stop (Seconds (endTime));

    serverApps.Get(0)->TraceConnectWithoutContext("Rx", MakeBoundCallback (&CallbackSinks::RxSink, stream));

    // Trigger data transfer
    Ptr<ExponentialRandomVariable> ftpArrivals = CreateObject<ExponentialRandomVariable> ();
    ftpArrivals->SetAttribute ("Mean", DoubleValue (1/lambda));

    double firstSend = ftpArrivals->GetValue () + startTime;
    if (firstSend < endTime)
    {
      Simulator::Schedule (Seconds (firstSend), &StartFileTransfer, clientApps, ftpArrivals, endTime);
      NS_LOG_INFO ("First file transmission scheduled at " << firstSend + Simulator::Now ().GetSeconds ());
    }

  }

  void
  SimulationConfig::StartFileTransfer (ApplicationContainer clientApps, Ptr<ExponentialRandomVariable> ftpArrivals, double endTime)
  {
    Ptr<FileTransferApplication> ftpApp = DynamicCast<FileTransferApplication> (clientApps.Get (0));
    NS_ASSERT (ftpApp);
    ftpApp->SendFile ();

    double nextSend = ftpArrivals->GetValue ();
    if (nextSend + Simulator::Now ().GetSeconds () < endTime)
    {
      Simulator::Schedule (Seconds (nextSend), &StartFileTransfer, clientApps, ftpArrivals, endTime);
      NS_LOG_INFO ("Next file transmission scheduled at " << nextSend + Simulator::Now ().GetSeconds ());
    }
    else
    {
      NS_LOG_INFO ("Not enough time for further transmissions");
    }
  }

  void
  SimulationConfig::InstallRmcatApps (bool nada, Ptr<Node> sender, Ptr<Node> receiver, uint16_t port, float initBw, float minBw, float maxBw, float startTime, float stopTime)
  {
      Ptr<RmcatSender> sendApp = CreateObject<RmcatSender> ();
      Ptr<RmcatReceiver> recvApp = CreateObject<RmcatReceiver> ();
      sender->AddApplication (sendApp);
      receiver->AddApplication (recvApp);

      Ptr<Ipv4> ipv4 = receiver->GetObject<Ipv4> ();
      Ipv4Address receiverIp = ipv4->GetAddress (1, 0).GetLocal ();
      sendApp->Setup (receiverIp, port); // initBw, minBw, maxBw);

      const auto fps = 25.;
      auto innerCodec = new syncodecs::StatisticsCodec{fps};
      auto codec = new syncodecs::ShapedPacketizer{innerCodec, DEFAULT_PACKET_SIZE};
      sendApp->SetCodec (std::shared_ptr<syncodecs::Codec>{codec});

      recvApp->Setup (port);

      sendApp->SetStartTime (Seconds (startTime));
      sendApp->SetStopTime (Seconds (stopTime));

      recvApp->SetStartTime (Seconds (startTime));
      recvApp->SetStopTime (Seconds (stopTime));
  }

  void
  SimulationConfig::SetupDashApplication (Ptr<Node> clientNode, Ptr<Node> serverNode, uint32_t port, uint8_t videoId,  double startTime, double stopTime, Ptr<OutputStreamWrapper> stream)
  {
    Ptr<Ipv4> ipv4 = serverNode->GetObject<Ipv4> ();
    Ipv4Address receiverIp = ipv4->GetAddress (1, 0).GetLocal ();

    DashClientHelper client("ns3::TcpSocketFactory", InetSocketAddress(receiverIp, port), "ns3::DashClient");
    //client.SetAttribute ("MaxBytes", UintegerValue (maxBytes));
    client.SetAttribute("VideoId", UintegerValue(videoId)); // VideoId should be positive
    client.SetAttribute("TargetDt", TimeValue(Seconds(35.0))); // The target time difference between receiving and playing a frame.
    client.SetAttribute("window", TimeValue(Time("10s")));  // The window for measuring the average throughput (Time).
    ApplicationContainer clientApp = client.Install(clientNode);
    clientApp.Start(Seconds(startTime));
    clientApp.Stop(Seconds(stopTime));

    DashServerHelper server("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer serverApps = server.Install(serverNode);
    serverApps.Start(Seconds(startTime));
    serverApps.Stop(Seconds(stopTime));

    serverApps.Get(0)->TraceConnectWithoutContext("Rx", MakeBoundCallback (&CallbackSinks::RxSink, stream));
  }

  void
  SimulationConfig::SetTracesPath (std::string filePath)
  {
    Config::SetDefault("ns3::MmWaveBearerStatsCalculator::DlRlcOutputFilename", StringValue(filePath + "DlRlcStats.txt"));
    Config::SetDefault("ns3::MmWaveBearerStatsCalculator::UlRlcOutputFilename", StringValue(filePath + "UlRlcStats.txt"));
    Config::SetDefault("ns3::MmWaveBearerStatsCalculator::DlPdcpOutputFilename", StringValue(filePath + "DlPdcpStats.txt"));
    Config::SetDefault("ns3::MmWaveBearerStatsCalculator::UlPdcpOutputFilename", StringValue(filePath + "UlPdcpStats.txt"));
    Config::SetDefault("ns3::MmWavePhyRxTrace::OutputFilename", StringValue(filePath + "RxPacketTrace.txt"));
    Config::SetDefault("ns3::LteRlcAm::BufferSizeFilename", StringValue(filePath + "RlcAmBufferSize.txt"));
  }

  void
  RandomBuildings::CreateRandomBuildings (double streetWidth, double blockSize, double maxXAxis, double maxYAxis, uint32_t numBlocks)
  {
    /* Create the building */
  	 double maxObstacleSize = blockSize - streetWidth;

  	 std::vector<Ptr<Building> > buildingVector;
  	 std::list<Box>  m_previousBlocks;

  	 for(uint32_t buildingIndex = 0; buildingIndex < numBlocks; buildingIndex++)
  	 {
  		 Ptr < Building > building;
  		 building = Create<Building> ();
  		 /* returns a vecotr where:
  		 * position [0]: coordinates for x min
  		 * position [1]: coordinates for x max
  		 * position [2]: coordinates for y min
  		 * position [3]: coordinates for y max
  		 */

  		 std::pair<Box, std::list<Box>> pairBuildings = RandomBuildings::GenerateBuildingBounds(0, maxXAxis-maxObstacleSize, 0, maxYAxis-maxObstacleSize, maxObstacleSize, m_previousBlocks);
  		 m_previousBlocks = std::get<1>(pairBuildings);
  	 	 Box box = std::get<0>(pairBuildings);

  		 Ptr<UniformRandomVariable> randomBuildingZ = CreateObject<UniformRandomVariable>();
  		 randomBuildingZ->SetAttribute("Min",DoubleValue(1.6));
  		 randomBuildingZ->SetAttribute("Max",DoubleValue(50));
  		 double buildingHeight = randomBuildingZ->GetValue();
       NS_LOG_INFO ("Building height " << buildingHeight << "\n");

  		 building->SetBoundaries (Box(box.xMin, box.xMax,
  																	 box.yMin,  box.yMax,
  																	 0.0, buildingHeight));

  		 building->SetNRoomsX(1);
  		 building->SetNRoomsY(1);
  		 building->SetNFloors(1);
  		 buildingVector.push_back(building);
  	 }
  		/* END Create the building */
  }

  void
  CallbackSinks::RxSink (Ptr<OutputStreamWrapper> stream, Ptr<const Packet> packet, const Address &from)
  {
    *stream->GetStream () << Simulator::Now ().GetSeconds () << "\t" << packet->GetSize() << std::endl;
  }

  std::pair<Box, std::list<Box>>
  RandomBuildings::GenerateBuildingBounds(double xMin, double xMax, double yMin, double yMax, double maxBuildSize, std::list<Box> m_previousBlocks )
  {

    Ptr<UniformRandomVariable> xMinBuilding = CreateObject<UniformRandomVariable>();
    xMinBuilding->SetAttribute("Min",DoubleValue(xMin));
    xMinBuilding->SetAttribute("Max",DoubleValue(xMax-1)); // 1 m is the minimum size

    Ptr<UniformRandomVariable> yMinBuilding = CreateObject<UniformRandomVariable>();
    yMinBuilding->SetAttribute("Min",DoubleValue(yMin));
    yMinBuilding->SetAttribute("Max",DoubleValue(yMax-1)); // 1 m is the minimum size

    Box box;
    uint32_t attempt = 0;
    do
    {
      NS_ASSERT_MSG(attempt < 100, "Too many failed attempts to position non-overlapping buildings. Maybe area too small or too many buildings?");
      box.xMin = xMinBuilding->GetValue();

      Ptr<UniformRandomVariable> xMaxBuilding = CreateObject<UniformRandomVariable>();
      xMaxBuilding->SetAttribute("Min",DoubleValue(box.xMin + 1)); // 1 m is the minimum size
      xMaxBuilding->SetAttribute("Max",DoubleValue(box.xMin + maxBuildSize));
      box.xMax = xMaxBuilding->GetValue();

      box.yMin = yMinBuilding->GetValue();

      Ptr<UniformRandomVariable> yMaxBuilding = CreateObject<UniformRandomVariable>();
      yMaxBuilding->SetAttribute("Min",DoubleValue(box.yMin + 1)); // 1 m is the minimum size
      yMaxBuilding->SetAttribute("Max",DoubleValue(box.yMin + maxBuildSize));
      box.yMax = yMaxBuilding->GetValue();

      ++attempt;
    }
    while (OverlapWithAnyPrevious (box, m_previousBlocks));


    NS_LOG_INFO("Building in coordinates (" << box.xMin << " , " << box.yMin << ") and ("  << box.xMax << " , " << box.yMax <<
      ") accepted after " << attempt << " attempts");
    m_previousBlocks.push_back(box);
    std::pair<Box, std::list<Box>> pairReturn = std::make_pair(box,m_previousBlocks);
    return pairReturn;
  }

  bool
  RandomBuildings::AreOverlapping(Box a, Box b)
  {
    return !((a.xMin > b.xMax) || (b.xMin > a.xMax) || (a.yMin > b.yMax) || (b.yMin > a.yMax) );
  }

  bool
  RandomBuildings::OverlapWithAnyPrevious(Box box, std::list<Box> m_previousBlocks)
  {
    for (std::list<Box>::iterator it = m_previousBlocks.begin(); it != m_previousBlocks.end(); ++it)
    {
      if (AreOverlapping(*it,box))
      {
        return true;
      }
    }
    return false;
  }

  void
  RandomBuildings::PrintGnuplottableBuildingListToFile (std::string filename)
  {
    std::ofstream outFile;
    outFile.open (filename.c_str (), std::ios_base::out | std::ios_base::trunc);
    if (!outFile.is_open ())
      {
        NS_LOG_ERROR ("Can't open file " << filename);
        return;
      }

  	outFile << "set xrange [0:100]" << std::endl;
  	outFile << "set yrange [0:100]" << std::endl;
  	outFile << "unset key" << std::endl;
  	outFile << "set grid" << std::endl;

    uint32_t index = 0;
    for (BuildingList::Iterator it = BuildingList::Begin (); it != BuildingList::End (); ++it)
      {
        ++index;
        Box box = (*it)->GetBoundaries ();
        outFile << "set object " << index
                << " rect from " << box.xMin  << "," << box.yMin
                << " to "   << box.xMax  << "," << box.yMax
                //<< " height " << box.zMin << "," << box.zMax
                << " front fs empty "
                << std::endl;
      }
  }


} // end namespace ns3
