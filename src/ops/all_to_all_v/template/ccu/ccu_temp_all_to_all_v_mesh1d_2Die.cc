/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_temp_all_to_all_v_mesh1d_2Die.h"
#include "kernel/ccu_kernel_all_to_all_v_mesh1d_2die.h"

namespace ops_hccl {
CcuTempAlltoAllVMesh1D2Die::CcuTempAlltoAllVMesh1D2Die(const OpParam &param, RankId rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    for (u32 i = 0; i < subCommRanks_.size(); i++) {
        for (u32 j = 0; j < subCommRanks_[i].size(); j++) {
            HCCL_INFO("subCommRanks_[%u][%u]=%u", i, j, subCommRanks_[i][j]);
        }
    }

    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        myRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempAlltoAllVMesh1D2Die::~CcuTempAlltoAllVMesh1D2Die()
{
}

HcclResult CcuTempAlltoAllVMesh1D2Die::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(subCommRanks_.size() != 1 || subCommRanks_[0].empty(),
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][CalcRes] Invalid subCommRanks[%u] or subCommRanks empty.",
            subCommRanks_.size()), HcclResult::HCCL_E_INTERNAL);

    HCCL_DEBUG("[CcuTempAlltoAllVMesh1D2Die][CalcRes] rankSize[%u] subCommRanks0[%u].", templateRankSize_,
        subCommRanks_[0].size());

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));
    HCCL_INFO("channelDescs size[%u]", channelDescs.size());

    CHK_RET(PartitionChannels(comm, channelDescs, rankIdToChannelDesc_));

    uint32_t slaveThreadNum = kernelCount_ - 1;
    resourceRequest.notifyNumOnMainThread = slaveThreadNum;
    resourceRequest.slaveThreadNum = slaveThreadNum;
    resourceRequest.notifyNumPerThread.assign(slaveThreadNum, 1);

    resourceRequest.channels.emplace_back(channelDescs);
    HCCL_INFO("resourceRequest.channels[%d]", resourceRequest.channels.size());

    resourceRequest.ccuKernelNum.push_back(kernelCount_);

    for (uint32_t i = 0; i < kernelCount_; i++) {
        CcuKernelInfo kernelInfo;
        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
            return std::make_unique<CcuKernelAllToAllVMesh1D2Die>(arg);
        };
        auto kernelArg = std::make_shared<CcuKernelArgAllToAllVMesh1D2Die>(myRank_, param, subCommRanks_,
            kernelWithMyRank_[i], kernelRankGroup_[i], is2Plus6_, closPeers_, kernelType_[i]);
        kernelInfo.kernelArg = kernelArg;
        kernelInfo.channels = kernelChannels_[i];
        resourceRequest.ccuKernelInfos.emplace_back(kernelInfo);
        HCCL_DEBUG("[CcuTempAlltoAllVMesh1D2Die][CalcRes] kernel[%u], channels=%llu, withMyRank=%u, kernelType=%u",
            i, kernelChannels_[i].size(), kernelWithMyRank_[i], kernelType_[i]);
    }

    CHK_RET(SaveCacheCtx(comm, param));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs,
                                                         std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
    (void) channelDescs;

    std::map<uint32_t, std::vector<HcclChannelDesc>> singleChByDie;
    std::map<uint32_t, std::vector<HcclChannelDesc>> multiChByDie;

    for (auto& rankToChannels : rankIdToChannelDesc) {
        u32 remoteRank = rankToChannels.first;
        std::vector<HcclChannelDesc>& channelList = rankToChannels.second;
        bool isMulti = channelList.size() > 1;
        if (isMulti) {
            is2Plus6_ = true;
            closPeers_.insert(remoteRank);
        }
        for (const auto& channel : channelList) {
            uint32_t dieId = 0;
            CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
            (isMulti ? multiChByDie : singleChByDie)[dieId].emplace_back(channel);
        }
    }

    auto fillKernel = [this](uint32_t kernelIdx, const std::vector<HcclChannelDesc>& channels) {
        for (const auto& ch : channels) {
            kernelChannels_[kernelIdx].emplace_back(ch);
            kernelRankGroup_[kernelIdx].push_back(ch.remoteRank);
        }
    };

    if (is2Plus6_) {
        kernelCount_ = 3;
        fullmeshDieId_ = singleChByDie.begin()->first;
        fillKernel(KERNEL_FULLMESH, singleChByDie[fullmeshDieId_]);
        kernelRankGroup_[KERNEL_FULLMESH].push_back(myRank_);
        for (auto& pair : multiChByDie) {
            fillKernel(pair.first == fullmeshDieId_ ? KERNEL_CLOS_MINOR : KERNEL_CLOS_MAJOR, pair.second);
        }
    } else {
        kernelCount_ = 2;
        auto it0 = singleChByDie.begin();
        auto it1 = std::next(it0);
        if (it0->second.size() > it1->second.size()) {
            std::swap(it0, it1);
        }
        fillKernel(KERNEL_FULLMESH, it0->second);
        kernelRankGroup_[KERNEL_FULLMESH].push_back(myRank_);
        fillKernel(KERNEL_CLOS_MAJOR, it1->second);
    }

    HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][PartitionChannels] Rank[%d], is2Plus6[%d], kernelCount[%u], "
        "fullmeshRankGroup[%zu], closMajorRankGroup[%zu], closMinorRankGroup[%zu].",
        myRank_, is2Plus6_, kernelCount_, kernelRankGroup_[KERNEL_FULLMESH].size(),
        kernelRankGroup_[KERNEL_CLOS_MAJOR].size(), kernelRankGroup_[KERNEL_CLOS_MINOR].size());

    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAlltoAllVMesh1D2Die::SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::SaveCacheCtx(HcclComm comm, const OpParam &param)
{
    Mesh2DieCacheCtx cacheCtx;
    cacheCtx.dieNum = DIE_NUM;
    cacheCtx.is2Plus6 = is2Plus6_;
    cacheCtx.kernelCount = kernelCount_;
    for (uint32_t i = 0; i < MAX_KERNEL_NUM_2DIE; i++) {
        cacheCtx.rankGroup[i] = kernelRankGroup_[i];
    }
    cacheCtx.closPeers = closPeers_;

    std::vector<char> buf = cacheCtx.Serialize();

    char cacheTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(cacheTag, sizeof(cacheTag), sizeof(cacheTag) - 1, "%s_mesh2die", param.algTag);
    CHK_PRT_RET(ret <= 0,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][SaveCacheCtx] failed to fill cacheTag"), HCCL_E_INTERNAL);

    void *ctxPtr = nullptr;
    uint64_t ctxSize = 0;
    HcclResult getRet = HcclEngineCtxGet(comm, cacheTag, CommEngine::COMM_ENGINE_CPU_TS, &ctxPtr, &ctxSize);
    if (getRet == HCCL_SUCCESS && ctxPtr != nullptr && ctxSize == buf.size()) {
        CHK_SAFETY_FUNC_RET(memcpy_s(ctxPtr, ctxSize, buf.data(), buf.size()));
        HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][SaveCacheCtx] updated existing cacheCtx, size[%zu]", buf.size());
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(HcclEngineCtxCreate(comm, cacheTag, CommEngine::COMM_ENGINE_CPU_TS, buf.size(), &ctxPtr));

    CHK_SAFETY_FUNC_RET(memcpy_s(ctxPtr, buf.size(), buf.data(), buf.size()));

    HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][SaveCacheCtx] created and saved cacheCtx, size[%zu], "
        "is2Plus6[%d] kernelCount[%u] closPeers[%zu]",
        buf.size(), is2Plus6_, kernelCount_, closPeers_.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::LoadCacheCtx(const OpParam &param, Mesh2DieCacheCtx &cacheCtx)
{
    char cacheTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(cacheTag, sizeof(cacheTag), sizeof(cacheTag) - 1, "%s_mesh2die", param.algTag);
    CHK_PRT_RET(ret <= 0,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][LoadCacheCtx] failed to fill cacheTag"), HCCL_E_INTERNAL);

    void *ctxPtr = nullptr;
    uint64_t ctxSize = 0;
    HcclComm comm = static_cast<HcclComm>(param.hcclComm);
    HcclResult getRet = HcclEngineCtxGet(comm, cacheTag, CommEngine::COMM_ENGINE_CPU_TS, &ctxPtr, &ctxSize);
    CHK_PRT_RET(getRet != HCCL_SUCCESS,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][LoadCacheCtx] HcclEngineCtxGet failed, ret=%d", getRet),
        getRet);
    CHK_PRT_RET(ctxPtr == nullptr || ctxSize == 0,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][LoadCacheCtx] cache ctx is null or empty"),
        HcclResult::HCCL_E_INTERNAL);

    CHK_RET(cacheCtx.Deserialize(static_cast<const char *>(ctxPtr), static_cast<size_t>(ctxSize)));

    HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][LoadCacheCtx] loaded cacheCtx, size[%llu], "
        "is2Plus6[%d] kernelCount[%u] closPeers[%zu]",
        ctxSize, cacheCtx.is2Plus6, cacheCtx.kernelCount, cacheCtx.closPeers.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllVMesh1D2Die::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
    TemplateResource& templateResource)
{
    CHK_PRT_RET(subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][KernelRun] subCommRanks empty."), HcclResult::HCCL_E_INTERNAL);

    buffInfo_ = templateDataParams.buffInfo;
    CHK_PRT_RET(buffInfo_.inputPtr == nullptr || buffInfo_.outputPtr == nullptr,
        HCCL_ERROR("[CcuTempAlltoAllVMesh1D2Die][KernelRun] Rank[%d] input[%#llx] or output[%#llx] is null",
            myRank_, buffInfo_.inputPtr, buffInfo_.outputPtr),
        HcclResult::HCCL_E_PTR);

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr);
    HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][KernelRun] begin. Rank[%d], input[%#llx/%#llx], output[%#llx/%#llx], "
        "sendType[%d], recvType[%d]", myRank_, inputAddr, param.inputPtr, outputAddr, param.outputPtr,
        param.all2AllVDataDes.sendType, param.all2AllVDataDes.recvType);

    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    Mesh2DieCacheCtx cacheCtx;
    CHK_RET(LoadCacheCtx(param, cacheCtx));

    uint32_t kernelCount = cacheCtx.kernelCount;
    uint32_t subThreadCount = kernelCount - 1;

    HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][KernelRun] Rank[%d], kernelCount[%u], subThreadCount[%u], "
        "is2Plus6[%d], token[%#llx].",
        myRank_, kernelCount, subThreadCount, cacheCtx.is2Plus6, token);

    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1,
        templateResource.threads.begin() + 1 + subThreadCount);
    std::vector<u32> notifyIdxMainToSub(subThreadCount, 0);
    CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));

    for (uint32_t i = 0; i < kernelCount; i++) {
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllToAllVMesh1D2Die>(
            inputAddr, outputAddr, token, localSendRecvInfo_);
        void *taskArgPtr = static_cast<void *>(taskArg.get());
        HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][KernelRun] kernel[%u] rankGroupSize[%zu]",
            i, cacheCtx.rankGroup[i].size());
        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[i],
            templateResource.ccuKernels[i], taskArgPtr));
    }

    std::vector<u32> notifyIdxSubToMain(subThreadCount);
    for (uint32_t i = 0; i < subThreadCount; i++) {
        notifyIdxSubToMain[i] = i;
    }
    CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));

    HCCL_INFO("[CcuTempAlltoAllVMesh1D2Die][KernelRun] end. Rank[%d], kernelCount[%u].", myRank_, kernelCount);

    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl
