/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_to_all_v_mesh1d_2die.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int CKE_IDX_0    = 0;
constexpr int POST_SYNC_ID = 3;

CcuKernelAllToAllVMesh1D2Die::CcuKernelAllToAllVMesh1D2Die(const hcomm::CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllToAllVMesh1D2Die *kernelArg = dynamic_cast<const CcuKernelArgAllToAllVMesh1D2Die *>(&arg);
    channels_ = kernelArg->channels;

    rankId_ = kernelArg->rankId_;
    withMyRank_ = kernelArg->withMyRank_;
    rankGroup_ = kernelArg->rankGroup_;
    is2Plus6_ = kernelArg->is2Plus6_;
    closPeers_ = kernelArg->closPeers_;
    kernelType_ = kernelArg->kernelType_;

    localId_ = channels_.size();
    peerSize_ = channels_.size() + (withMyRank_ ? 1 : 0);

    HCCL_INFO("[CcuKernelAllToAllVMesh1D2Die] rankId[%u], peerSize[%u], withMyRank[%u], is2Plus6[%d], kernelType[%u]",
        rankId_, peerSize_, withMyRank_, is2Plus6_, kernelType_);
}

HcclResult CcuKernelAllToAllVMesh1D2Die::InitResources()
{
    CHK_PRT_RET(channels_.empty(),
        HCCL_ERROR("[CcuKernelAllToAllVMesh1D2Die] RankId[%u] channels is empty!", rankId_),
        HcclResult::HCCL_E_INTERNAL);

    AllocGoResource(CCU_MS_LOCAL_COPY_LOOP_COUNT, LOCAL_COPY_MS_PER_LOOP);
    xnMaxTransportSize_ = CreateVariable();
    xnMaxTransportGoSize_ = CreateGroupOpSize();
    xnMaxTransportSize_ = MAX_TRANSPORT_SIZE;
    auto xnMaxTransportGoSize = CalGoSize(MAX_TRANSPORT_SIZE);
    xnMaxTransportGoSize_.addrOffset = xnMaxTransportGoSize[GO_ADDR_OFFSET_IDX];
    xnMaxTransportGoSize_.loopParam = xnMaxTransportGoSize[GO_LOOP_PARAM_IDX];
    xnMaxTransportGoSize_.parallelParam = xnMaxTransportGoSize[GO_PARALLEL_PARAM_IDX];
    xnMaxTransportGoSize_.residual = xnMaxTransportGoSize[GO_RESIDUAL_IDX];

    input_ = CreateVariable();

    for (uint32_t peerId = 0; peerId < channels_.size(); peerId++) {
        HCCL_DEBUG("[CcuKernelAllToAllVMesh1D2Die] RankId[%u], PeerId[%u]", rankId_, peerId);
        hcomm::CcuRep::Variable output;
        CHK_RET(CreateVariable(channels_[peerId], OUTPUT_XN_ID, &output));
        output_.emplace_back(output);
        hcomm::CcuRep::Variable token;
        CHK_RET(CreateVariable(channels_[peerId], TOKEN_XN_ID, &token));
        token_.emplace_back(token);
    }
    output_.emplace_back(CreateVariable());
    token_.emplace_back(CreateVariable());

    sendRecvInfo_.resize(peerSize_);
    for (uint64_t peerId = 0; peerId < peerSize_; peerId++) {
        sendRecvInfo_[peerId].sendOffset = CreateVariable();
        sendRecvInfo_[peerId].recvOffset = CreateVariable();
        sendRecvInfo_[peerId].sendTailSize = CreateVariable();
        sendRecvInfo_[peerId].sendTailGoSize = CreateGroupOpSize();
        sendRecvInfo_[peerId].sendLoopNum = CreateVariable();
    }

    for (uint16_t i = 0; i < channels_.size(); i++) {
        src_.emplace_back(CreateLocalAddr());
        dst_.emplace_back(CreateRemoteAddr());
    }

    localSrc_ = CreateLocalAddr();
    localDst_ = CreateLocalAddr();

    curSendTailSize_ = CreateVariable();
    curSendTailGoSize_ = CreateGroupOpSize();

    xnConst1_ = CreateVariable();
    completedRankCount_ = CreateVariable();

    const uint32_t eventNum = (peerSize_ + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    events_.resize(eventNum);
    for (uint32_t i = 0; i < eventNum; i++) {
        events_[i] = CreateCompletedEvent();
    }

    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllToAllVMesh1D2Die::LoadArgs()
{
    Load(input_);
    Load(output_[localId_]);
    Load(token_[localId_]);

    for (uint64_t peerId = 0; peerId < peerSize_; peerId++) {
        Load(sendRecvInfo_[peerId].sendOffset);
        Load(sendRecvInfo_[peerId].recvOffset);
        Load(sendRecvInfo_[peerId].sendTailSize);
        Load(sendRecvInfo_[peerId].sendTailGoSize);
        Load(sendRecvInfo_[peerId].sendLoopNum);
    }
}

void CcuKernelAllToAllVMesh1D2Die::ExchangeInfoSync()
{
    hcomm::CcuRep::Variable tempDst = CreateVariable();
    for (u32 peerId = 0; peerId < channels_.size(); peerId++) {
        tempDst = output_[localId_];
        tempDst += sendRecvInfo_[peerId].recvOffset;
        NotifyRecord(channels_[peerId], CKE_IDX_0, OUTPUT_XN_ID, tempDst, 1 << OUTPUT_XN_ID);
        NotifyRecord(channels_[peerId], CKE_IDX_0, TOKEN_XN_ID, token_[localId_], 1 << TOKEN_XN_ID);
    }
    uint32_t waitBits = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (u32 peerId = 0; peerId < channels_.size(); peerId++) {
        NotifyWait(channels_[peerId], CKE_IDX_0, waitBits);
    }
}

void CcuKernelAllToAllVMesh1D2Die::PostSync()
{
    for (const auto &channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (u32 peerId = 0; peerId < channels_.size(); peerId++) {
        NotifyWait(channels_[peerId], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelAllToAllVMesh1D2Die::DoAll2AllVMultiLoop()
{
    completedRankCount_ = 0;
    xnConst1_ = 1;
    CCU_WHILE(completedRankCount_ != peerSize_) {
        HCCL_DEBUG("[CcuKernelAllToAllVMesh1D2Die] Algorithm loops[%u].", peerSize_);
        LoopStep();
    }
}

void CcuKernelAllToAllVMesh1D2Die::WriteToDstOutput(uint32_t peerId)
{
    const uint16_t eventIdx = peerId / BIT_NUM_PER_CKE;
    const uint16_t rankMask = 1 << (peerId % BIT_NUM_PER_CKE);
    events_[eventIdx].SetMask(rankMask);

    CCU_IF(sendRecvInfo_[peerId].sendLoopNum == UINT64_MAX)
    {
        RecordEvent(events_[eventIdx]);
    }

    CCU_IF(sendRecvInfo_[peerId].sendLoopNum != UINT64_MAX)
    {
        CCU_IF(sendRecvInfo_[peerId].sendLoopNum == UINT64_MAX - 1)
        {
            curSendTailSize_ = sendRecvInfo_[peerId].sendTailSize;
            CCU_IF(curSendTailSize_ == 0)
            {
                RecordEvent(events_[eventIdx]);
            }
            CCU_IF(curSendTailSize_ != 0)
            {
                WriteNb(channels_[peerId], dst_[peerId], src_[peerId], curSendTailSize_, events_[eventIdx]);
            }
            completedRankCount_ += xnConst1_;
        }
        CCU_IF(sendRecvInfo_[peerId].sendLoopNum != UINT64_MAX - 1)
        {
            WriteNb(channels_[peerId], dst_[peerId], src_[peerId], xnMaxTransportSize_, events_[eventIdx]);
            dst_[peerId].addr += xnMaxTransportSize_;
            src_[peerId].addr += xnMaxTransportSize_;
        }
        sendRecvInfo_[peerId].sendLoopNum += xnConst1_;
    }
}

void CcuKernelAllToAllVMesh1D2Die::GroupCopyToDstOutput(uint32_t peerId)
{
    const uint16_t eventIdx = peerId / BIT_NUM_PER_CKE;
    const uint16_t rankMask = 1 << (peerId % BIT_NUM_PER_CKE);
    events_[eventIdx].SetMask(rankMask);

    CCU_IF(sendRecvInfo_[peerId].sendLoopNum == UINT64_MAX)
    {
        RecordEvent(events_[eventIdx]);
    }

    CCU_IF(sendRecvInfo_[peerId].sendLoopNum != UINT64_MAX)
    {
        CCU_IF(sendRecvInfo_[peerId].sendLoopNum == UINT64_MAX - 1)
        {
            curSendTailSize_ = sendRecvInfo_[peerId].sendTailSize;
            curSendTailGoSize_ = sendRecvInfo_[peerId].sendTailGoSize;
            CCU_IF(curSendTailSize_ == 0)
            {
                RecordEvent(events_[eventIdx]);
            }
            CCU_IF(curSendTailSize_ != 0)
            {
                GroupCopy(localDst_, localSrc_, curSendTailGoSize_);
                RecordEvent(events_[eventIdx]);
            }
            completedRankCount_ += xnConst1_;
        }
        CCU_IF(sendRecvInfo_[peerId].sendLoopNum != UINT64_MAX - 1)
        {
            GroupCopy(localDst_, localSrc_, xnMaxTransportGoSize_);
            RecordEvent(events_[eventIdx]);
            localDst_.addr += xnMaxTransportSize_;
            localSrc_.addr += xnMaxTransportSize_;
        }
        sendRecvInfo_[peerId].sendLoopNum += xnConst1_;
    }
}

void CcuKernelAllToAllVMesh1D2Die::CalcGroupSrcDst()
{
    for (uint32_t peerId = 0; peerId < channels_.size(); peerId++) {
        src_[peerId].addr = input_;
        src_[peerId].addr += sendRecvInfo_[peerId].sendOffset;
        src_[peerId].token = token_[peerId];
        dst_[peerId].addr = output_[peerId];
        dst_[peerId].token = token_[peerId];
    }

    if (withMyRank_) {
        localSrc_.addr = input_;
        localSrc_.addr += sendRecvInfo_[localId_].sendOffset;
        localSrc_.token = token_[localId_];
        localDst_.addr = output_[localId_];
        localDst_.addr += sendRecvInfo_[localId_].recvOffset;
        localDst_.token = token_[localId_];
    }
}

void CcuKernelAllToAllVMesh1D2Die::LoopStep()
{
    for (uint32_t peerId = 0; peerId < channels_.size(); peerId++) {
        WriteToDstOutput(peerId);
    }

    if (withMyRank_) {
        GroupCopyToDstOutput(localId_);
    }

    const uint32_t eventNum = (peerSize_ + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    for (uint32_t i = 0; i < eventNum; i++) {
        uint16_t eventMask;
        if (i == eventNum - 1) {
            if (peerSize_ % BIT_NUM_PER_CKE == 0) {
                eventMask = (1 << BIT_NUM_PER_CKE) - 1;
            } else {
                eventMask = (1 << (peerSize_ % BIT_NUM_PER_CKE)) - 1;
            }
        } else {
            eventMask = (1 << BIT_NUM_PER_CKE) - 1;
        }
        events_[i].SetMask(eventMask);
        WaitEvent(events_[i]);
    }
}

HcclResult CcuKernelAllToAllVMesh1D2Die::Algorithm()
{
    HCCL_INFO("[CcuKernelAllToAllVMesh1D2Die] Algorithm Init Begins. RankId[%u]", rankId_);
    CHK_RET(InitResources());
    LoadArgs();

    HCCL_INFO("[CcuKernelAllToAllVMesh1D2Die] Algorithm Begins. RankId[%u]", rankId_);

    ExchangeInfoSync();
    CalcGroupSrcDst();
    DoAll2AllVMultiLoop();
    PostSync();

    HCCL_INFO("[CcuKernelAllToAllVMesh1D2Die] Algorithm Ends. RankId[%u]", rankId_);

    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllToAllVMesh1D2Die::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllToAllVMesh1D2Die *taskArg = dynamic_cast<const CcuTaskArgAllToAllVMesh1D2Die *>(&arg);
    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;

    std::vector<uint64_t> taskParams = {inputAddr, outputAddr, tokenInfo};

    const auto &sendRecvInfo = taskArg->localSendRecvInfo_;
    for (auto peerId : rankGroup_) {
        uint64_t sendSize = sendRecvInfo.sendLength[peerId];
        uint64_t sendOffset = sendRecvInfo.sendOffset[peerId];
        uint64_t recvOffset = sendRecvInfo.recvOffset[peerId];

        if (is2Plus6_ && closPeers_.count(peerId) > 0) {
            uint64_t recvLength = sendRecvInfo.recvLength[peerId];
            uint64_t minorSendSize = sendSize * CLOS_RATIO_MINOR / CLOS_RATIO_TOTAL;
            uint64_t minorRecvSize = recvLength * CLOS_RATIO_MINOR / CLOS_RATIO_TOTAL;
            if (kernelType_ == KERNEL_CLOS_MAJOR) {
                sendOffset += minorSendSize;
                recvOffset += minorRecvSize;
                sendSize = sendSize - minorSendSize;
            } else if (kernelType_ == KERNEL_CLOS_MINOR) {
                sendSize = minorSendSize;
            }
        }

        const uint64_t floorLoopNum = sendSize / MAX_TRANSPORT_SIZE;
        uint64_t sendLoopNum = UINT64_MAX - 1 - floorLoopNum;
        uint64_t sendTailSize = sendSize - floorLoopNum * MAX_TRANSPORT_SIZE;
        auto sendTailGoSize = CalGoSize(sendTailSize);
        taskParams.push_back(sendOffset);
        taskParams.push_back(recvOffset);
        taskParams.push_back(sendTailSize);
        taskParams.insert(taskParams.cend(), sendTailGoSize.cbegin(), sendTailGoSize.cend());
        taskParams.push_back(sendLoopNum);
        HCCL_DEBUG("[CcuKernelAllToAllVMesh1D2Die][GeneArgs] RankId[%u] peer[%u] "
            "sendOffset[%llu] recvOffset[%llu] sendTailSize[%llu] sendLoopNum[%llu]",
            rankId_, peerId, sendOffset, recvOffset, sendTailSize, sendLoopNum);
    }

    HCCL_INFO("[CcuKernelAllToAllVMesh1D2Die][GeneArgs] RankId[%u], inputAddr[%#llx], outputAddr[%#llx], "
        "args[%zu]", rankId_, inputAddr, outputAddr, taskParams.size());

    return taskParams;
}

}
