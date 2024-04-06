/*
 * Copyright (c) 2016 Georgia Institute of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __CPU_GARNET_SYNTHETIC_TRAFFIC_HH__
#define __CPU_GARNET_SYNTHETIC_TRAFFIC_HH__

#include <set>

#include "base/statistics.hh"
#include "mem/port.hh"
#include "params/GarnetSyntheticTraffic.hh"
#include "sim/clocked_object.hh"
#include "sim/eventq.hh"
#include "sim/sim_exit.hh"
#include "sim/sim_object.hh"
#include "sim/stats.hh"
#include "mem/ruby/network/garnet/dnn_data_tag.hh"

enum TrafficType {BIT_COMPLEMENT_ = 0,
                  BIT_REVERSE_ = 1,
                  BIT_ROTATION_ = 2,
                  NEIGHBOR_ = 3,
                  SHUFFLE_ = 4,
                  TORNADO_ = 5,
                  TRANSPOSE_ = 6,
                  UNIFORM_RANDOM_ = 7,
                  DNN_ = 8,
                  NUM_TRAFFIC_PATTERNS_};

class Packet;
class GarnetSyntheticTraffic : public ClockedObject
{
  public:
  int cal_cycles;
    int packets_to_send ;
    int send_dst;
    int send_cmd_dst;
    int packets_sent;
    int cpu_status;
    int num_packet_wait;
    int wait_data_tag;
    int send_data_tag;
    int pkt_num_wgt_recv;
    int pkt_num_act_recv;
    int pkt_num_out_recv;
    int downstream_id;
    int cycles_caled;
    int total_packet_recv_previous;
    int total_packet_recv_previous_wgt;
    int total_packet_recv_previous_act;
    int total_packet_recv_previous_out;
    int cpu_work_stats;
    std::string current_task_line;
    int current_line_num;
    int Repeat_Start_line;
    int cur_pic;
    int num_recv_cmd_packet;
    int get_task(int id,int line_num);
    int check_downstream(int id);
    void update_cur_pic(int id, int cur_pic_id);
    void tell_mem_send_data(std::string src_mem_index,std::string num_wait_packets,int id);
    int time_cal  ;
    int time_wait  ;
    int time_wait_cmd  ;
    int time_send ;
    
    typedef GarnetSyntheticTrafficParams Params;
    GarnetSyntheticTraffic(const Params *p);

    void init() override;

    // main simulation loop (one cycle)
    void tick();

    Port &getPort(const std::string &if_name,
                  PortID idx=InvalidPortID) override;

    /**
     * Print state of address in memory system via PrintReq (for
     * debugging).
     */
    void printAddr(Addr a);

  protected:
    EventFunctionWrapper tickEvent;

    class CpuPort : public RequestPort
    {
        GarnetSyntheticTraffic *tester;

      public:

        CpuPort(const std::string &_name, GarnetSyntheticTraffic *_tester)
            : RequestPort(_name, _tester), tester(_tester)
        { }

      protected:

        virtual bool recvTimingResp(PacketPtr pkt);

        virtual void recvReqRetry();
    };

    CpuPort cachePort;

    class GarnetSyntheticTrafficSenderState : public Packet::SenderState
    {
      public:
        /** Constructor. */
        GarnetSyntheticTrafficSenderState(uint8_t *_data)
            : data(_data)
        { }

        // Hold onto data pointer
        uint8_t *data;
    };

    PacketPtr retryPkt;
    unsigned size;
    int id;

    std::map<std::string, TrafficType> trafficStringToEnum;

    unsigned blockSizeBits;

    Tick noResponseCycles;

    int numDestinations;
    Tick simCycles;
    int numPacketsMax;
    int numPacketsSent;
    int singleSender;
    int singleDest;

    std::string trafficType; // string
    TrafficType traffic; // enum from string
    double injRate;
    int injVnet;
    int if_debug;
    std::string dnn_task;
    int precision;

    const Cycles responseLimit;

    RequestorID requestorId;

    void completeRequest(PacketPtr pkt);
     int recv_packets(int id, int data_tag);

    void generatePkt(int send_dst);
    void sendPkt(PacketPtr pkt);
    void initTrafficType();

    void doRetry();

    friend class MemCompleteEvent;
};

#endif // __CPU_GARNET_SYNTHETIC_TRAFFIC_HH__
