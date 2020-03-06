# MilliSlice - Enabling RAN Slicing Through Carrier Aggregation in mmWave Cellular Networks #

The ever increasing number of connected devices and of new, heterogeneous mobile use cases implies that 5G cellular systems will face demanding technical challenges. For example, Ultra-Reliable Low-Latency Communication (URLLC) and enhanced Mobile Broadband (eMBB) scenarios present orthogonal Quality of Service (QoS) requirements that the 5G aims to satisfy with a unified Radio Access Network (RAN) design. Network slicing and mmWave communications have been identified as possible enablers for 5G, as they provide, respectively, the necessary scalability and flexibility to adapt the network to each specific use case environment, and low latency and multi-gigabit-per-second wireless links, which tap into a vast, currently unused portion of the spectrum. The optimization and integration of these technologies is still an open research challenge, which requires innovations at different layers of the protocol stack. This paper proposes to combine them in a RAN slicing framework for mmWaves, based on carrier aggregation. Notably, we introduce MilliSlice, a cross-carrier scheduling policy that exploits the diversity of the carriers and maximizes their utilization, thus simultaneously guaranteeing high throughput for the eMBB slices and low latency, high reliability for the URLLC flows.

## Notes ##

This code is based on the ns3-mmwave module https://github.com/nyuwireless-unipd/ns3-mmwave.git.

## Authors ##

 - Matteo Pagin <mattpagg@gmail.com>
 - Francesco Agostini <francesco.agostini.4@studenti.unipd.it>
 - Tommaso Zugno <tommasozugno@gmail.com>
 - Michele Polese <michele.polese@gmail.com>

## License ##

This software is licensed under the terms of the GNU GPLv2, as like as ns-3. See the LICENSE file for more details.
