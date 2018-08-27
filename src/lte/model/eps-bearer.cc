/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */


#include "eps-bearer.h"

#include <ns3/fatal-error.h>


namespace ns3 {


GbrQosInformation::GbrQosInformation ()
  : gbrDl (0),
    gbrUl (0),
    mbrDl (0),
    mbrUl (0)
{
}

AllocationRetentionPriority::AllocationRetentionPriority ()
  : priorityLevel (0),
    preemptionCapability (false),
    preemptionVulnerability (false)
{
}

EpsBearer::EpsBearer ()
  : qci (NGBR_VIDEO_TCP_DEFAULT)
{
}

EpsBearer::EpsBearer (Qci x)
  : qci (x)
{
}

EpsBearer::EpsBearer (Qci x, struct GbrQosInformation y)
  : qci (x), gbrQosInfo (y)
{
}

bool
EpsBearer::IsGbr () const
{
  // 3GPP 23.203 Section 5.7.4
  switch (qci)
    {
    case GBR_CONV_VOICE:
    case GBR_CONV_VIDEO:
    case GBR_GAMING:
    case GBR_NON_CONV_VIDEO:
    case GBR_MISSION_CRITICAL_PTT:
    case GBR_NON_MISSION_CRITICAL_PTT:
    case GBR_MISSION_CRITICAL_VIDEO:
    case GBR_V2X:
    case DCGBR_REMOTE_CONTROL:
    case DCGBR_INTELLIGENT_TRANSPORT_SYS:
    case DCGBR_INTELLIGENT_TRANSPORT_SYS_2:
    case DCGBR_DISCRETE_AUTOMATION:
    case DCGBR_DISCRETE_AUTOMATION_2:
      return true;
    case NGBR_IMS:
    case NGBR_VIDEO_TCP_OPERATOR:
    case NGBR_VOICE_VIDEO_GAMING:
    case NGBR_VIDEO_TCP_PREMIUM:
    case NGBR_VIDEO_TCP_DEFAULT:
    case NGBR_MISSION_CRITICAL_DELAY_SENSITIVE:
    case NGBR_MISSION_CRITICAL_DATA:
    case NGBR_V2X:
    case NGBR_LOW_LAT_EMBB:
      return false;
    default:
      NS_FATAL_ERROR ("unknown QCI value " << qci);
      return false;
    }
}

uint8_t
EpsBearer::GetPriority () const
{
  // 3GPP 23.203 Section 5.7.4
  switch (qci)
    {
    case GBR_CONV_VOICE:
      return 20;
    case GBR_CONV_VIDEO:
      return 40;
    case GBR_GAMING:
      return 30;
    case GBR_NON_CONV_VIDEO:
      return 50;
    case GBR_MISSION_CRITICAL_PTT:
      return 7;
    case GBR_NON_MISSION_CRITICAL_PTT:
      return 20;
    case GBR_MISSION_CRITICAL_VIDEO:
      return 15;
    case GBR_V2X:
      return 25;
    case NGBR_IMS:
      return 10;
    case NGBR_VIDEO_TCP_OPERATOR:
      return 60;
    case NGBR_VOICE_VIDEO_GAMING:
      return 70;
    case NGBR_VIDEO_TCP_PREMIUM:
      return 80;
    case NGBR_VIDEO_TCP_DEFAULT:
      return 90;
    case NGBR_MISSION_CRITICAL_DELAY_SENSITIVE:
      return 5;
    case NGBR_MISSION_CRITICAL_DATA:
      return 55;
    case NGBR_V2X:
      return 65;
    case NGBR_LOW_LAT_EMBB:
      return 68;
    case DCGBR_REMOTE_CONTROL:
      return 11;
    case DCGBR_INTELLIGENT_TRANSPORT_SYS:
      return 12;
    case DCGBR_INTELLIGENT_TRANSPORT_SYS_2:
      return 13;
    case DCGBR_DISCRETE_AUTOMATION:
      return 19;
    case DCGBR_DISCRETE_AUTOMATION_2:
      return 22;
    default:
      NS_FATAL_ERROR ("unknown QCI value " << qci);
      return 0;
    }
}

uint16_t
EpsBearer::GetPacketDelayBudgetMs () const
{
  // 3GPP 23.203 Section 5.7.4
  switch (qci)
    {
    case GBR_CONV_VOICE:
      return 100;
    case GBR_CONV_VIDEO:
      return 150;
    case GBR_GAMING:
      return 50;
    case GBR_NON_CONV_VIDEO:
      return 300;
    case GBR_MISSION_CRITICAL_PTT:
      return 75;
    case GBR_NON_MISSION_CRITICAL_PTT:
      return 100;
    case GBR_MISSION_CRITICAL_VIDEO:
      return 100;
    case GBR_V2X:
      return 50;
    case NGBR_IMS:
      return 100;
    case NGBR_VIDEO_TCP_OPERATOR:
      return 300;
    case NGBR_VOICE_VIDEO_GAMING:
      return 100;
    case NGBR_VIDEO_TCP_PREMIUM:
      return 300;
    case NGBR_VIDEO_TCP_DEFAULT:
      return 300;
    case NGBR_MISSION_CRITICAL_DELAY_SENSITIVE:
      return 60;
    case NGBR_MISSION_CRITICAL_DATA:
      return 200;
    case NGBR_V2X:
      return 50;
    case NGBR_LOW_LAT_EMBB:
      return 10;
    case DCGBR_REMOTE_CONTROL:
      return 5;
    case DCGBR_INTELLIGENT_TRANSPORT_SYS:
      return 10;
    case DCGBR_INTELLIGENT_TRANSPORT_SYS_2:
      return 20;
    case DCGBR_DISCRETE_AUTOMATION:
      return 10;
    case DCGBR_DISCRETE_AUTOMATION_2:
      return 10;
    default:
      NS_FATAL_ERROR ("unknown QCI value " << qci);
      return 0;
    }
}

double
EpsBearer::GetPacketErrorLossRate () const
{
  // 3GPP 23.203 Section 5.7.4
  switch (qci)
    {
    case GBR_CONV_VOICE:
      return 1.0e-2;
    case GBR_CONV_VIDEO:
      return 1.0e-3;
    case GBR_GAMING:
      return 1.0e-3;
    case GBR_NON_CONV_VIDEO:
      return 1.0e-6;
    case GBR_MISSION_CRITICAL_PTT:
      return 1.0e-2;
    case GBR_NON_MISSION_CRITICAL_PTT:
      return 1.0e-2;
    case GBR_MISSION_CRITICAL_VIDEO:
      return 1.0e-3;
    case GBR_V2X:
      return 1.0e-2;
    case NGBR_IMS:
      return 1.0e-6;
    case NGBR_VIDEO_TCP_OPERATOR:
      return 1.0e-6;
    case NGBR_VOICE_VIDEO_GAMING:
      return 1.0e-3;
    case NGBR_VIDEO_TCP_PREMIUM:
      return 1.0e-6;
    case NGBR_VIDEO_TCP_DEFAULT:
      return 1.0e-6;
    case NGBR_MISSION_CRITICAL_DELAY_SENSITIVE:
      return 1.0e-6;
    case NGBR_MISSION_CRITICAL_DATA:
      return 1.0e-6;
    case NGBR_V2X:
      return 1.0e-2;
    case NGBR_LOW_LAT_EMBB:
      return 1.0e-6;
    case DCGBR_REMOTE_CONTROL:
      return 1.0e-5;
    case DCGBR_INTELLIGENT_TRANSPORT_SYS:
      return 1.0e-5;
    case DCGBR_INTELLIGENT_TRANSPORT_SYS_2:
      return 1.0e-5;
    case DCGBR_DISCRETE_AUTOMATION:
      return 1.0e-4;
    case DCGBR_DISCRETE_AUTOMATION_2:
      return 1.0e-4;
    default:
      NS_FATAL_ERROR ("unknown QCI value " << qci);
      return 0;
    }
}

uint16_t
EpsBearer::GetDefaultAveragingWindow () const
{
  // 3GPP 23.203 Section 5.7.4
  switch (qci)
    {
    case GBR_CONV_VOICE:
      return 2000;
    case GBR_CONV_VIDEO:
      return 2000;
    case GBR_GAMING:
      return 2000;
    case GBR_NON_CONV_VIDEO:
      return 2000;
    case GBR_MISSION_CRITICAL_PTT:
      return 2000;
    case GBR_NON_MISSION_CRITICAL_PTT:
      return 2000;
    case GBR_MISSION_CRITICAL_VIDEO:
      return 2000;
    case GBR_V2X:
      return 2000;
    case NGBR_IMS:
    case NGBR_VIDEO_TCP_OPERATOR:
    case NGBR_VOICE_VIDEO_GAMING:
    case NGBR_VIDEO_TCP_PREMIUM:
    case NGBR_VIDEO_TCP_DEFAULT:
    case NGBR_MISSION_CRITICAL_DELAY_SENSITIVE:
    case NGBR_MISSION_CRITICAL_DATA:
    case NGBR_V2X:
    case NGBR_LOW_LAT_EMBB:
      return 0;
    case DCGBR_REMOTE_CONTROL:
      return 2000;
    case DCGBR_INTELLIGENT_TRANSPORT_SYS:
      return 2000;
    case DCGBR_INTELLIGENT_TRANSPORT_SYS_2:
      return 2000;
    case DCGBR_DISCRETE_AUTOMATION:
      return 2000;
    case DCGBR_DISCRETE_AUTOMATION_2:
      return 2000;
    default:
      NS_FATAL_ERROR ("unknown QCI value " << qci);
      return 0;
    }
}

uint16_t
EpsBearer::GetDefaultMaximumDataBurstVolume () const
{
  // 3GPP 23.203 Section 5.7.4
  switch (qci)
    {
    case GBR_CONV_VOICE:
    case GBR_CONV_VIDEO:
    case GBR_GAMING:
    case GBR_NON_CONV_VIDEO:
    case GBR_MISSION_CRITICAL_PTT:
    case GBR_NON_MISSION_CRITICAL_PTT:
    case GBR_MISSION_CRITICAL_VIDEO:
    case GBR_V2X:
    case NGBR_IMS:
    case NGBR_VIDEO_TCP_OPERATOR:
    case NGBR_VOICE_VIDEO_GAMING:
    case NGBR_VIDEO_TCP_PREMIUM:
    case NGBR_VIDEO_TCP_DEFAULT:
    case NGBR_MISSION_CRITICAL_DELAY_SENSITIVE:
    case NGBR_MISSION_CRITICAL_DATA:
    case NGBR_V2X:
    case NGBR_LOW_LAT_EMBB:
      return 0;
    case DCGBR_REMOTE_CONTROL:
      return 160;
    case DCGBR_INTELLIGENT_TRANSPORT_SYS:
      return 320;
    case DCGBR_INTELLIGENT_TRANSPORT_SYS_2:
      return 640;
    case DCGBR_DISCRETE_AUTOMATION:
      return 255;
    case DCGBR_DISCRETE_AUTOMATION_2:
      return 1358;
    default:
      NS_FATAL_ERROR ("unknown QCI value " << qci);
      return 0;
    }
}

} // namespace ns3
