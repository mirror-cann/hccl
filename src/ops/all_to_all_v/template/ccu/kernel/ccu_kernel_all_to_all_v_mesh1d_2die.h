/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH1D_2DIE_H
#define HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH1D_2DIE_H

#include <vector>
#include <set>
#include "utils.h"
#include "template_utils.h"
#include "ccu_kernel.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "ccu_temp_all_to_all_v_mesh1d_2Die.h"

namespace ops_hccl {

using RankId = u32;

class CcuKernelArgAllToAllVMesh1D2Die : public hcomm::CcuKernelArg {
public:
    CcuKernelArgAllToAllVMesh1D2Die(uint32_t rankId, const OpParam &opParam,
        const std::vector<std::vector<RankId>> &subCommRanks, bool withMyRank,
        const std::vector<RankId> &rankGroup, bool is2Plus6,
        const std::set<RankId> &closPeers, uint32_t kernelType)
        : rankId_(rankId), opParam_(opParam), subCommRanks_(subCommRanks), withMyRank_(withMyRank),
          rankGroup_(rankGroup), is2Plus6_(is2Plus6), closPeers_(closPeers), kernelType_(kernelType) {}

    ~CcuKernelArgAllToAllVMesh1D2Die() override {}

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        GenerateCcuKernelSignature(signature, "CcuKernelArgAllToAllVMesh1D2Die", opParam_, subCommRanks_);
        return signature;
    }

    uint32_t rankId_;
    OpParam opParam_;
    std::vector<std::vector<RankId>> subCommRanks_;
    bool withMyRank_;
    std::vector<RankId> rankGroup_;
    bool is2Plus6_;
    std::set<RankId> closPeers_;
    uint32_t kernelType_;
};

class CcuTaskArgAllToAllVMesh1D2Die : public hcomm::CcuTaskArg {
public:
    explicit CcuTaskArgAllToAllVMesh1D2Die(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
        const A2ASendRecvInfo& localSendRecvInfo)
        : inputAddr_(inputAddr), outputAddr_(outputAddr), token_(token), localSendRecvInfo_(localSendRecvInfo) {}

    uint64_t inputAddr_;
    uint64_t outputAddr_;
    uint64_t token_;
    A2ASendRecvInfo localSendRecvInfo_;
};

class CcuKernelAllToAllVMesh1D2Die : public CcuKernelAlgBase {
public:
    CcuKernelAllToAllVMesh1D2Die(const hcomm::CcuKernelArg &arg);
    ~CcuKernelAllToAllVMesh1D2Die() override {}

    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

protected:
    struct A2AVSingleSendRecvInfo {
        CcuRep::Variable sendOffset;
        CcuRep::Variable recvOffset;
        CcuRep::Variable sendTailSize;
        GroupOpSize      sendTailGoSize;
        CcuRep::Variable sendLoopNum;
    };

private:
    HcclResult InitResources();
    void LoadArgs();
    void ExchangeInfoSync();
    void PostSync();
    void DoAll2AllVMultiLoop();
    void LoopStep();
    void CalcGroupSrcDst();
    void WriteToDstOutput(uint32_t peerId);
    void GroupCopyToDstOutput(uint32_t peerId);

    const uint32_t GO_ADDR_OFFSET_IDX = 0;
    const uint32_t GO_LOOP_PARAM_IDX = 1;
    const uint32_t GO_PARALLEL_PARAM_IDX = 2;
    const uint32_t GO_RESIDUAL_IDX = 3;

    const uint64_t MAX_TRANSPORT_SIZE = UB_MAX_TRANS_SIZE;
    static constexpr uint16_t BIT_NUM_PER_CKE = 16;

    uint32_t rankId_{0};
    bool withMyRank_{false};
    uint32_t localId_{0};
    uint32_t peerSize_{0};
    std::vector<RankId> rankGroup_;
    std::vector<ChannelHandle> channels_;

    bool is2Plus6_{false};
    std::set<RankId> closPeers_;
    uint32_t kernelType_{0};

    hcomm::CcuRep::Variable input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;

    hcomm::CcuRep::LocalAddr localSrc_;
    hcomm::CcuRep::LocalAddr localDst_;

    std::vector<hcomm::CcuRep::LocalAddr> src_;
    std::vector<hcomm::CcuRep::RemoteAddr> dst_;

    hcomm::CcuRep::Variable xnConst1_;
    hcomm::CcuRep::Variable completedRankCount_;
    hcomm::CcuRep::Variable xnMaxTransportSize_;
    GroupOpSize xnMaxTransportGoSize_;
    hcomm::CcuRep::Variable curSendTailSize_;
    GroupOpSize curSendTailGoSize_;
    std::vector<A2AVSingleSendRecvInfo> sendRecvInfo_;

    std::vector<hcomm::CcuRep::CompletedEvent> events_;
};

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_TO_ALL_V_MESH1D_2DIE_H
