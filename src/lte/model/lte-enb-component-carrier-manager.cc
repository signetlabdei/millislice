/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Danilo Abrignani
 * Copyright (c) 2018, University of Padova, Dep. of Information Engineering, SIGNET lab.
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
 * Author: Danilo Abrignani <danilo.abrignani@unibo.it>
 *
 * Modified by: Tommaso Zugno <tommasozugno@gmail.com>
 *								 Integration of Carrier Aggregation for the mmWave module
 */

#include "lte-enb-component-carrier-manager.h"
#include <ns3/log.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteEnbComponentCarrierManager");
NS_OBJECT_ENSURE_REGISTERED (LteEnbComponentCarrierManager);


LteEnbComponentCarrierManager::LteEnbComponentCarrierManager ()
{

}

LteEnbComponentCarrierManager::~LteEnbComponentCarrierManager ()
{
}

TypeId
LteEnbComponentCarrierManager::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::LteEnbComponentCarrierManager")
    .SetParent<Object> ()
    .SetGroupName("Lte")
  ;
  return tid;
}

void
LteEnbComponentCarrierManager::DoDispose ()
{
}

void
LteEnbComponentCarrierManager::SetLteCcmRrcSapUser (LteCcmRrcSapUser* s)
{
  NS_LOG_FUNCTION (this << s);
  m_ccmRrcSapUser = s;
}

LteCcmRrcSapProvider*
LteEnbComponentCarrierManager::GetLteCcmRrcSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_ccmRrcSapProvider;
}

LteMacSapProvider*
LteEnbComponentCarrierManager::GetLteMacSapProvider ()
{
  NS_LOG_FUNCTION (this);
  return m_macSapProvider;
}

LteCcmMacSapUser*
LteEnbComponentCarrierManager::GetLteCcmMacSapUser ()
{
  NS_LOG_FUNCTION (this);
  return m_ccmMacSapUser;
}

bool
LteEnbComponentCarrierManager::SetMacSapProvider (uint8_t componentCarrierId, LteMacSapProvider* sap)
{
  NS_LOG_FUNCTION (this);
  bool res = false;
  std::map <uint8_t, LteMacSapProvider*>::iterator it = m_macSapProvidersMap.find (componentCarrierId);
  if ((uint16_t) componentCarrierId > m_noOfComponentCarriers)
    {
      NS_FATAL_ERROR ("Inconsistent componentCarrierId or you didn't call SetNumberOfComponentCarriers before calling this method");
    }
  if (it != m_macSapProvidersMap.end ())
    {
      NS_FATAL_ERROR ("Tried to allocated an existing componentCarrierId");
    }
  else
    {
      m_macSapProvidersMap.insert (std::pair<uint8_t, LteMacSapProvider*>(componentCarrierId, sap));
      res = true;
    }
  return res;

}

bool
LteEnbComponentCarrierManager::SetCcmMacSapProviders (uint8_t componentCarrierId, LteCcmMacSapProvider* sap)
{
  NS_LOG_FUNCTION (this);
  bool res = false;
  std::map< uint8_t,LteCcmMacSapProvider*>::iterator it =  m_ccmMacSapProviderMap.find (componentCarrierId);

  if (it == m_ccmMacSapProviderMap.end ())
    {
      m_ccmMacSapProviderMap.insert (std::pair <uint8_t,LteCcmMacSapProvider*> (componentCarrierId, sap));
    }

  res = true;
  return res;

}

void
LteEnbComponentCarrierManager::SetNumberOfComponentCarriers (uint16_t noOfComponentCarriers)
{
  NS_LOG_FUNCTION (this);
  m_noOfComponentCarriers = noOfComponentCarriers;
}

void
LteEnbComponentCarrierManager::SetRrc (const Ptr<LteEnbRrc> rrc)
{
  m_rrc = rrc;
}

void
LteEnbComponentCarrierManager::SetBandwidthMap (std::map<uint8_t, double> bandwidthMap)
{
  m_bandwidthMap = bandwidthMap;
}

void
LteEnbComponentCarrierManager::SetDrbCcMap (std::map<uint16_t, uint8_t> drbCcMap)
{
  m_drbCcMap = drbCcMap;
}

} // end of namespace ns3
