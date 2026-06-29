/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#include <limits>
#include <set>
#include <vector>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/p_relu_grad_reduce_tiling_data.h"
#include "../op_kernel/p_relu_grad_reduce_tiling_key.h"

namespace optiling {
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t IDX_WEIGHTS = 2;
constexpr int64_t IDX_UPDATES = 3;
constexpr int64_t IDX_DA = 0;
constexpr uint32_t REDUCE_MODE_ALL = 0;
constexpr uint32_t REDUCE_MODE_NCHW_CHANNEL = 1;
constexpr uint32_t REDUCE_MODE_ND_WEIGHT_SHAPE = 2;
constexpr uint32_t REDUCE_MODE_NC1HWC0 = 3;
constexpr uint32_t REDUCE_MODE_SHARED_PARTIAL = 4;
constexpr uint32_t REDUCE_MODE_SHARED_FINAL = 5;
constexpr uint32_t DTYPE_MODE_FP16 = 0;
constexpr uint32_t DTYPE_MODE_FP32 = 1;
constexpr uint32_t DTYPE_MODE_BF16 = 2;
constexpr uint32_t DA_ALIGN = 16;
constexpr uint32_t FP32_DA_ALIGN = 8;

struct PReluGradReduceCompileInfo {};

// 获取真实的硬件核心数
// 本算子是纯 Vector 计算（无 Cube），分离架构芯片（如 910B）上要用 Vector 核数(AIV)，
// 不能用 Cube 核数(AIC) —— 910B 系列 AIC:AIV 通常是 1:2（如 20:40 / 24:48），
// 之前用 GetCoreNumAic() 会把 coreNum 提前钳在较小的 Cube 核数上，导致 blockDim 始终偏小。
static uint32_t GetActualCoreNum(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t coreNum = aivNum;
    if (coreNum == 0) coreNum = aicNum;
    if (coreNum == 0) coreNum = ascendcPlatform.GetCoreNum();
    if (coreNum == 0) coreNum = 20; // 910B 默认保底
    OP_LOGD(context, "PReluGradReduce GetActualCoreNum: aicNum=%u, aivNum=%u, chosenCoreNum=%u",
        aicNum, aivNum, coreNum);
    return coreNum;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, const PReluGradReduceTilingData* tiling, uint32_t coreNum)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t rawSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t sysWorkspaceSize = (rawSysWorkspaceSize + 31) / 32 * 32;

    if (tiling->reduceMode == REDUCE_MODE_ALL) {
        // REDUCE_MODE_ALL 走真正的多核 map-reduce，需要两块 workspace：
        // 1) SyncAll 跨核同步区：要求 >= coreNum*32Bytes，kernel 侧会在用前清零
        // 2) 各核局部和区：coreNum 个 float，32B 对齐
        size_t syncBufSize = static_cast<size_t>(coreNum) * 32;
        size_t partialBufSize = ((static_cast<size_t>(coreNum) * sizeof(float) + 31) / 32) * 32;
        currentWorkspace[0] = sysWorkspaceSize + syncBufSize + partialBufSize;
    } else if (tiling->reduceMode == REDUCE_MODE_NCHW_CHANNEL && tiling->reduceSplit > 1) {
        // REDUCE_MODE_NCHW_CHANNEL 走 channel × reduceSlice 二阶段归约：
        // 1) SyncAll 跨核同步区：blockDim 个参与核，每核 32B
        // 2) partial sum 区：outputLength * reduceSplit 个 FP32 partial，32B 对齐
        size_t syncBufSize = static_cast<size_t>(tiling->blockDim) * 32;
        size_t partialCount = static_cast<size_t>(tiling->outputLength) * tiling->reduceSplit;
        size_t partialBufSize = ((partialCount * sizeof(float) + 31) / 32) * 32;
        currentWorkspace[0] = sysWorkspaceSize + syncBufSize + partialBufSize;
    } else if (tiling->reduceMode == REDUCE_MODE_NC1HWC0 && tiling->reduceSplit > 1) {
        // REDUCE_MODE_NC1HWC0 走 C1 × reduceSlice 二阶段归约：
        // 1) SyncAll 跨核同步区：blockDim 个参与核，每核 32B
        // 2) partial sum 区：C1 * reduceSplit * C0 个 FP32 partial，32B 对齐
        size_t syncBufSize = static_cast<size_t>(tiling->blockDim) * 32;
        size_t partialCount = static_cast<size_t>(tiling->c1) * tiling->reduceSplit * tiling->c0;
        size_t partialBufSize = ((partialCount * sizeof(float) + 31) / 32) * 32;
        currentWorkspace[0] = sysWorkspaceSize + syncBufSize + partialBufSize;
    } else if (tiling->reduceMode == REDUCE_MODE_NC1HWC0) {
        uint32_t partialSumWorkspaceSize = coreNum * DA_ALIGN * sizeof(float);
        currentWorkspace[0] = sysWorkspaceSize + partialSumWorkspaceSize;
    } else {
        currentWorkspace[0] = sysWorkspaceSize;
    }
    return ge::GRAPH_SUCCESS;
}

static uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return ((value + align - 1) / align) * align;
}

static uint32_t GetKernelBlockDim(const PReluGradReduceTilingData* tiling, uint32_t coreNum)
{
    if (tiling->reduceMode == REDUCE_MODE_ALL) {
        uint32_t elemPerCore = tiling->dtypeMode == DTYPE_MODE_FP32 ? 2048 : 4096;
        uint32_t needCores = (tiling->totalLength + elemPerCore - 1) / elemPerCore;
        if (needCores >= coreNum) return coreNum;
        if (needCores >= 8) return 8;
        if (needCores >= 4) return 4;
        if (needCores >= 2) return 2;
        return 1;
    }

    if (tiling->reduceMode == REDUCE_MODE_NCHW_CHANNEL && tiling->outputLength < coreNum) {
        uint32_t reduceElems = tiling->n * tiling->h * tiling->w;
        if (reduceElems == 0) reduceElems = 1;
        uint32_t elemPerCore = tiling->dtypeMode == DTYPE_MODE_FP32 ? 2048 : 4096;
        uint32_t maxReduceChunks = (reduceElems + elemPerCore - 1) / elemPerCore;
        if (maxReduceChunks == 0) maxReduceChunks = 1;

        uint32_t reduceSplit = coreNum / tiling->outputLength;
        if (reduceSplit > maxReduceChunks) reduceSplit = maxReduceChunks;
        if (reduceSplit > 1) {
            uint32_t totalCores = tiling->outputLength * reduceSplit;
            if (totalCores > coreNum) totalCores = coreNum;
            if (totalCores == 0) totalCores = 1;
            return totalCores;
        }
    }

    if (tiling->reduceMode == REDUCE_MODE_NC1HWC0 && tiling->c0 == DA_ALIGN && tiling->c1 < coreNum) {
        uint32_t reduceElems = tiling->n * tiling->h * tiling->w;
        if (reduceElems == 0) reduceElems = 1;
        uint32_t elemPerCore = tiling->dtypeMode == DTYPE_MODE_FP32 ? 2048 : 4096;
        uint32_t maxReduceChunks = (reduceElems + elemPerCore - 1) / elemPerCore;
        if (maxReduceChunks == 0) maxReduceChunks = 1;

        uint32_t c1Chunks = tiling->c1 == 0 ? 1 : tiling->c1;
        uint32_t reduceSplit = coreNum / c1Chunks;
        if (reduceSplit > maxReduceChunks) reduceSplit = maxReduceChunks;
        if (reduceSplit > 1) {
            uint32_t totalCores = c1Chunks * reduceSplit;
            if (totalCores > coreNum) totalCores = coreNum;
            if (totalCores == 0) totalCores = 1;
            return totalCores;
        }
    }

    uint32_t align = tiling->dtypeMode == DTYPE_MODE_FP32 ? FP32_DA_ALIGN : DA_ALIGN;
    uint32_t outputChunks = (tiling->outputLength + align - 1) / align;
    if (outputChunks == 0) outputChunks = 1;
    if (outputChunks >= coreNum) return coreNum;

    if (tiling->reduceMode != REDUCE_MODE_NC1HWC0) {
        if (outputChunks >= 8) return 8;
        if (outputChunks >= 4) return 4;
        if (outputChunks >= 2) return 2;
        return 1;
    }

    uint32_t reduceElems = tiling->n * tiling->h * tiling->w;
    if (reduceElems == 0) reduceElems = 1;
    uint32_t elemPerCore = tiling->dtypeMode == DTYPE_MODE_FP32 ? 2048 : 4096;
    uint32_t maxReduceChunks = (reduceElems + elemPerCore - 1) / elemPerCore;
    if (maxReduceChunks == 0) maxReduceChunks = 1;

    uint32_t nReduceChunks = coreNum / outputChunks;
    if (nReduceChunks > maxReduceChunks) nReduceChunks = maxReduceChunks;
    if (nReduceChunks == 0) nReduceChunks = 1;

    uint32_t totalCores = outputChunks * nReduceChunks;
    if (totalCores > coreNum) totalCores = coreNum;
    if (totalCores == 0) totalCores = 1;
    return totalCores;
}

static uint32_t GetReduceSplit(const PReluGradReduceTilingData* tiling)
{
    if (tiling->reduceMode == REDUCE_MODE_NCHW_CHANNEL && tiling->outputLength > 0) {
        uint32_t reduceSplit = tiling->blockDim / tiling->outputLength;
        return reduceSplit == 0 ? 1 : reduceSplit;
    }
    if (tiling->reduceMode == REDUCE_MODE_NC1HWC0 && tiling->c0 == DA_ALIGN && tiling->c1 > 0) {
        uint32_t reduceSplit = tiling->blockDim / tiling->c1;
        return reduceSplit == 0 ? 1 : reduceSplit;
    }
    return 1;
}

template <typename ShapeT>
static int64_t GetShapeSize(const ShapeT& shape)
{
    int64_t size = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        size *= shape.GetDim(i);
    }
    return size;
}

template <typename ShapeT>
static std::vector<int64_t> GetShapeVector(const ShapeT& shape)
{
    std::vector<int64_t> dims(shape.GetDimNum(), 1);
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        dims[i] = shape.GetDim(i);
    }
    return dims;
}

static ge::graphStatus CheckDimValue(gert::TilingContext* context, const std::vector<int64_t>& dims, const char* tensorName)
{
    for (auto dim : dims) {
        OP_CHECK_IF(dim <= 0,
            OP_LOGE(context, "%s shape dim should be positive, got %ld", tensorName, dim),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(dim > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
            OP_LOGE(context, "%s shape dim exceeds uint32 range, got %ld", tensorName, dim),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillCommonShape(PReluGradReduceTilingData* tiling,
    const std::vector<int64_t>& updatesShape, const std::vector<int64_t>& weightsShape)
{
    tiling->updatesRank = static_cast<uint32_t>(updatesShape.size());
    tiling->weightsRank = static_cast<uint32_t>(weightsShape.size());
    tiling->updatesShape0 = updatesShape.size() > 0 ? static_cast<uint32_t>(updatesShape[0]) : 1;
    tiling->updatesShape1 = updatesShape.size() > 1 ? static_cast<uint32_t>(updatesShape[1]) : 1;
    tiling->updatesShape2 = updatesShape.size() > 2 ? static_cast<uint32_t>(updatesShape[2]) : 1;
    tiling->updatesShape3 = updatesShape.size() > 3 ? static_cast<uint32_t>(updatesShape[3]) : 1;
    tiling->weightsShape0 = weightsShape.size() > 0 ? static_cast<uint32_t>(weightsShape[0]) : 1;
    tiling->weightsShape1 = weightsShape.size() > 1 ? static_cast<uint32_t>(weightsShape[1]) : 1;
    tiling->weightsShape2 = weightsShape.size() > 2 ? static_cast<uint32_t>(weightsShape[2]) : 1;
    return ge::GRAPH_SUCCESS;
}

static bool IsNc1hwc0Format(ge::Format format)
{
    return format == ge::FORMAT_NC1HWC0;
}

static ge::graphStatus CheckFormatSemantics(gert::TilingContext* context,
    const std::vector<int64_t>& updatesShape, const std::vector<int64_t>& weightsShape,
    ge::Format updatesFormat, ge::Format weightsFormat, ge::Format daFormat)
{
    const bool updatesNc1hwc0 = IsNc1hwc0Format(updatesFormat);
    const bool weightsNc1hwc0 = IsNc1hwc0Format(weightsFormat);
    const bool daNc1hwc0 = IsNc1hwc0Format(daFormat);
    const bool anyFormatNc1hwc0 = updatesNc1hwc0 || weightsNc1hwc0 || daNc1hwc0;

    if (anyFormatNc1hwc0) {
        OP_CHECK_IF(!updatesNc1hwc0 || !weightsNc1hwc0 || !daNc1hwc0,
            OP_LOGE(context, "NC1HWC0 branch requires updates, weights and da storage formats to all be NC1HWC0"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(updatesShape.size() != 5,
            OP_LOGE(context, "NC1HWC0 format expects updates rank 5, got %zu", updatesShape.size()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(weightsShape.size() != 2 && weightsShape.size() != 4 && weightsShape.size() != 5,
            OP_LOGE(context, "NC1HWC0 format expects weights rank 2, 4 or 5, got %zu", weightsShape.size()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    if (updatesShape.size() == 5) {
        OP_CHECK_IF(weightsShape.size() != 2 && weightsShape.size() != 4 && weightsShape.size() != 5,
            OP_LOGE(context, "Rank-5 ND updates expect weights rank 2, 4 or 5, got %zu", weightsShape.size()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillReduceMode(gert::TilingContext* context, PReluGradReduceTilingData* tiling,
    const std::vector<int64_t>& updatesShape, const std::vector<int64_t>& weightsShape)
{
    const bool weightShare = weightsShape.size() == 1 && weightsShape[0] == 1;
    if (weightShare) {
        tiling->reduceMode = REDUCE_MODE_ALL;
        tiling->n = 1; tiling->c = 1; tiling->h = 1; tiling->w = 1; tiling->c1 = 1; tiling->c0 = 1;
        return ge::GRAPH_SUCCESS;
    }

    if (updatesShape.size() == 5) {
        bool validNc1hwc0Weight = false;
        if (weightsShape.size() == 5) {
            validNc1hwc0Weight = weightsShape[1] == updatesShape[1] && weightsShape[4] == updatesShape[4];
        } else if (weightsShape.size() == 4) {
            validNc1hwc0Weight = weightsShape[0] == updatesShape[1] && weightsShape[3] == updatesShape[4];
        } else if (weightsShape.size() == 2) {
            validNc1hwc0Weight = weightsShape[0] == updatesShape[1] && weightsShape[1] == updatesShape[4];
        }
        OP_CHECK_IF(!validNc1hwc0Weight,
            OP_LOGE(context, "NC1HWC0 branch expects weights shape [C1,C0], [C1,1,1,C0] or [1,C1,1,1,C0]"),
            return ge::GRAPH_FAILED);
        tiling->reduceMode = REDUCE_MODE_NC1HWC0;
        tiling->n = static_cast<uint32_t>(updatesShape[0]);
        tiling->c = 1;
        tiling->h = static_cast<uint32_t>(updatesShape[2]);
        tiling->w = static_cast<uint32_t>(updatesShape[3]);
        tiling->c1 = static_cast<uint32_t>(updatesShape[1]);
        tiling->c0 = static_cast<uint32_t>(updatesShape[4]);
        return ge::GRAPH_SUCCESS;
    }

    if ((weightsShape.size() == updatesShape.size() - 1) && weightsShape.size() != 1) {
        OP_CHECK_IF(updatesShape.size() < 3 || updatesShape.size() > 4,
            OP_LOGE(context, "V9 ND weight-shape branch supports updates rank 3 or 4, got %zu", updatesShape.size()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(weightsShape.size() > 3,
            OP_LOGE(context, "V9 ND weight-shape branch supports weights rank <= 3, got %zu", weightsShape.size()),
            return ge::GRAPH_FAILED);
        tiling->reduceMode = REDUCE_MODE_ND_WEIGHT_SHAPE;
        tiling->n = 1; tiling->c = 1; tiling->h = 1; tiling->w = 1; tiling->c1 = 1; tiling->c0 = 1;
        return ge::GRAPH_SUCCESS;
    }

    if (updatesShape.size() >= 2 && updatesShape.size() <= 4 && weightsShape.size() == 1) {
        OP_CHECK_IF(updatesShape[1] != weightsShape[0],
            OP_LOGE(context, "ND/NCHW channel branch expects weights shape [C], got C %ld weight %ld",
                updatesShape[1], weightsShape[0]),
            return ge::GRAPH_FAILED);
        tiling->reduceMode = REDUCE_MODE_NCHW_CHANNEL;
        tiling->n = static_cast<uint32_t>(updatesShape[0]);
        tiling->c = static_cast<uint32_t>(updatesShape[1]);
        tiling->h = updatesShape.size() > 2 ? static_cast<uint32_t>(updatesShape[2]) : 1;
        tiling->w = updatesShape.size() > 3 ? static_cast<uint32_t>(updatesShape[3]) : 1;
        tiling->c1 = 1; tiling->c0 = 1;
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(context, "Unsupported PReluGradReduce shape branch, updates rank %zu, weights rank %zu",
        updatesShape.size(), weightsShape.size());
    return ge::GRAPH_FAILED;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, PReluGradReduceTilingData* tiling, uint32_t coreNum)
{
    auto updatesShapePtr = context->GetInputShape(IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesShapePtr);
    const auto& updatesStorageShape = updatesShapePtr->GetStorageShape();
    std::vector<int64_t> updatesShape = GetShapeVector(updatesStorageShape);
    OP_CHECK_IF(CheckDimValue(context, updatesShape, "updates") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "updates shape check failed"),
        return ge::GRAPH_FAILED);

    auto weightsShapePtr = context->GetInputShape(IDX_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsShapePtr);
    const auto& weightsStorageShape = weightsShapePtr->GetStorageShape();
    std::vector<int64_t> weightsShape = GetShapeVector(weightsStorageShape);
    OP_CHECK_IF(CheckDimValue(context, weightsShape, "weights") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "weights shape check failed"),
        return ge::GRAPH_FAILED);

    int64_t totalLength = GetShapeSize(updatesStorageShape);
    int64_t outputLength = GetShapeSize(weightsStorageShape);
    OP_CHECK_IF(totalLength <= 0 || totalLength > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        OP_LOGE(context, "updates shape size should be positive uint32, got %ld", totalLength),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputLength <= 0 || outputLength > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        OP_LOGE(context, "weights shape size should be positive uint32, got %ld", outputLength),
        return ge::GRAPH_FAILED);
    tiling->totalLength = static_cast<uint32_t>(totalLength);
    tiling->outputLength = static_cast<uint32_t>(outputLength);

    auto updatesDesc = context->GetInputDesc(IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesDesc);
    ge::DataType dataType = updatesDesc->GetDataType();
    ge::Format updatesFormat = updatesDesc->GetStorageFormat();
    auto weightsDesc = context->GetInputDesc(IDX_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsDesc);
    ge::Format weightsFormat = weightsDesc->GetStorageFormat();
    auto daDesc = context->GetOutputDesc(IDX_DA);
    OP_CHECK_NULL_WITH_CONTEXT(context, daDesc);
    ge::Format daFormat = daDesc->GetStorageFormat();
    OP_CHECK_IF(CheckFormatSemantics(context, updatesShape, weightsShape, updatesFormat, weightsFormat, daFormat) !=
        ge::GRAPH_SUCCESS,
        OP_LOGE(context, "format semantics check failed"),
        return ge::GRAPH_FAILED);

    if (dataType == ge::DT_FLOAT) {
        tiling->dtypeMode = DTYPE_MODE_FP32;
    } else if (dataType == ge::DT_BF16) {
        tiling->dtypeMode = DTYPE_MODE_BF16;
    } else {
        tiling->dtypeMode = DTYPE_MODE_FP16;
    }
    tiling->daPaddedLength = AlignUp(tiling->outputLength,
        tiling->dtypeMode == DTYPE_MODE_FP32 ? FP32_DA_ALIGN : DA_ALIGN);

    OP_CHECK_IF(FillCommonShape(tiling, updatesShape, weightsShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "FillCommonShape failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(FillReduceMode(context, tiling, updatesShape, weightsShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "FillReduceMode failed"),
        return ge::GRAPH_FAILED);

    // 写入运行时实际核数，供 kernel 侧 REDUCE_MODE_ALL 多核 map-reduce 使用
    tiling->coreNum = coreNum;
    // 使用动态硬件核数限制并发，防止 SyncAll 死锁
    tiling->blockDim = GetKernelBlockDim(tiling, coreNum);
    tiling->reduceSplit = GetReduceSplit(tiling);
    OP_LOGD(context,
        "PReluGradReduce tiling result: reduceMode=%u, totalLength=%u, outputLength=%u, "
        "coreNum=%u, blockDim=%u, reduceSplit=%u", tiling->reduceMode, tiling->totalLength, tiling->outputLength,
        coreNum, tiling->blockDim, tiling->reduceSplit);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PReluGradReduceTilingFunc(gert::TilingContext* context)
{
    PReluGradReduceTilingData* tiling = context->GetTilingData<PReluGradReduceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t rawSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    tiling->sysWorkspaceSize = static_cast<uint32_t>((rawSysWorkspaceSize + 31) / 32 * 32);

    // 提取当前设备的真实物理核数
    uint32_t coreNum = GetActualCoreNum(context);

    OP_CHECK_IF(GetShapeAttrsInfo(context, tiling, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context, tiling, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    context->SetBlockDim(tiling->blockDim);

    uint32_t tilingKey = P_RELU_GRAD_REDUCE_TILING_KEY_FP16;
    if (tiling->dtypeMode == DTYPE_MODE_FP32) {
        tilingKey = P_RELU_GRAD_REDUCE_TILING_KEY_FP32;
    } else if (tiling->dtypeMode == DTYPE_MODE_BF16) {
        tilingKey = P_RELU_GRAD_REDUCE_TILING_KEY_BF16;
    }

    context->SetTilingKey(GET_TPL_TILING_KEY(tilingKey));
    return ge::GRAPH_SUCCESS;
}

static void SetDtypeMode(PReluGradReduceTilingData* tiling, ge::DataType dataType)
{
    if (dataType == ge::DT_FLOAT) {
        tiling->dtypeMode = DTYPE_MODE_FP32;
    } else if (dataType == ge::DT_BF16) {
        tiling->dtypeMode = DTYPE_MODE_BF16;
    } else {
        tiling->dtypeMode = DTYPE_MODE_FP16;
    }
}

static void SetTilingKeyByDtype(gert::TilingContext* context, const PReluGradReduceTilingData* tiling)
{
    uint32_t tilingKey = P_RELU_GRAD_REDUCE_TILING_KEY_FP16;
    if (tiling->dtypeMode == DTYPE_MODE_FP32) {
        tilingKey = P_RELU_GRAD_REDUCE_TILING_KEY_FP32;
    } else if (tiling->dtypeMode == DTYPE_MODE_BF16) {
        tilingKey = P_RELU_GRAD_REDUCE_TILING_KEY_BF16;
    }
    context->SetTilingKey(GET_TPL_TILING_KEY(tilingKey));
}

static ge::graphStatus SetSysWorkspace(gert::TilingContext* context, PReluGradReduceTilingData* tiling)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t rawSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t sysWorkspaceSize = (rawSysWorkspaceSize + 31) / 32 * 32;
    tiling->sysWorkspaceSize = static_cast<uint32_t>(sysWorkspaceSize);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PReluGradReducePartialTilingFunc(gert::TilingContext* context)
{
    PReluGradReduceTilingData* tiling = context->GetTilingData<PReluGradReduceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto updatesShapePtr = context->GetInputShape(IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesShapePtr);
    const auto& updatesStorageShape = updatesShapePtr->GetStorageShape();
    std::vector<int64_t> updatesShape = GetShapeVector(updatesStorageShape);
    OP_CHECK_IF(CheckDimValue(context, updatesShape, "updates") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "updates shape check failed"),
        return ge::GRAPH_FAILED);

    auto weightsShapePtr = context->GetInputShape(IDX_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsShapePtr);
    const auto& weightsStorageShape = weightsShapePtr->GetStorageShape();
    std::vector<int64_t> weightsShape = GetShapeVector(weightsStorageShape);
    OP_CHECK_IF(CheckDimValue(context, weightsShape, "weights") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "weights shape check failed"),
        return ge::GRAPH_FAILED);

    int64_t totalLength = GetShapeSize(updatesStorageShape);
    OP_CHECK_IF(totalLength <= 0 || totalLength > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        OP_LOGE(context, "updates shape size should be positive uint32, got %ld", totalLength),
        return ge::GRAPH_FAILED);
    uint32_t nDim = updatesShape.empty() ? 1U : static_cast<uint32_t>(updatesShape[0]);

    auto updatesDesc = context->GetInputDesc(IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesDesc);
    SetDtypeMode(tiling, updatesDesc->GetDataType());

    tiling->reduceMode = REDUCE_MODE_SHARED_PARTIAL;
    tiling->totalLength = static_cast<uint32_t>(totalLength);
    tiling->outputLength = nDim;
    tiling->daPaddedLength = AlignUp(nDim, FP32_DA_ALIGN);
    tiling->n = nDim;
    tiling->c = updatesShape.size() > 1 ? static_cast<uint32_t>(updatesShape[1]) : 1;
    tiling->h = updatesShape.size() > 2 ? static_cast<uint32_t>(updatesShape[2]) : 1;
    tiling->w = updatesShape.size() > 3 ? static_cast<uint32_t>(updatesShape[3]) : 1;
    tiling->c1 = 1;
    tiling->c0 = 1;
    OP_CHECK_IF(FillCommonShape(tiling, updatesShape, weightsShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "FillCommonShape failed"),
        return ge::GRAPH_FAILED);

    uint32_t coreNum = GetActualCoreNum(context);
    tiling->coreNum = coreNum;
    tiling->blockDim = nDim < coreNum ? nDim : coreNum;
    if (tiling->blockDim == 0) tiling->blockDim = 1;
    tiling->reduceSplit = 1;

    OP_CHECK_IF(SetSysWorkspace(context, tiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetSysWorkspace error"),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(tiling->blockDim);
    SetTilingKeyByDtype(context, tiling);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PReluGradReduceFinalTilingFunc(gert::TilingContext* context)
{
    PReluGradReduceTilingData* tiling = context->GetTilingData<PReluGradReduceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto partialShapePtr = context->GetInputShape(IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context, partialShapePtr);
    const auto& partialStorageShape = partialShapePtr->GetStorageShape();
    std::vector<int64_t> partialShape = GetShapeVector(partialStorageShape);
    OP_CHECK_IF(CheckDimValue(context, partialShape, "partial") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "partial shape check failed"),
        return ge::GRAPH_FAILED);

    auto weightsShapePtr = context->GetInputShape(IDX_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsShapePtr);
    const auto& weightsStorageShape = weightsShapePtr->GetStorageShape();
    std::vector<int64_t> weightsShape = GetShapeVector(weightsStorageShape);
    OP_CHECK_IF(CheckDimValue(context, weightsShape, "weights") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "weights shape check failed"),
        return ge::GRAPH_FAILED);

    int64_t totalLength = GetShapeSize(partialStorageShape);
    int64_t outputLength = GetShapeSize(weightsStorageShape);
    OP_CHECK_IF(totalLength <= 0 || totalLength > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        OP_LOGE(context, "partial shape size should be positive uint32, got %ld", totalLength),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputLength <= 0 || outputLength > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        OP_LOGE(context, "weights shape size should be positive uint32, got %ld", outputLength),
        return ge::GRAPH_FAILED);

    auto weightsDesc = context->GetInputDesc(IDX_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsDesc);
    SetDtypeMode(tiling, weightsDesc->GetDataType());

    tiling->reduceMode = REDUCE_MODE_SHARED_FINAL;
    tiling->totalLength = static_cast<uint32_t>(totalLength);
    tiling->outputLength = static_cast<uint32_t>(outputLength);
    tiling->daPaddedLength = AlignUp(tiling->outputLength,
        tiling->dtypeMode == DTYPE_MODE_FP32 ? FP32_DA_ALIGN : DA_ALIGN);
    tiling->n = 1;
    tiling->c = 1;
    tiling->h = 1;
    tiling->w = 1;
    tiling->c1 = 1;
    tiling->c0 = 1;
    OP_CHECK_IF(FillCommonShape(tiling, partialShape, weightsShape) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "FillCommonShape failed"),
        return ge::GRAPH_FAILED);

    tiling->coreNum = 1;
    tiling->blockDim = 1;
    tiling->reduceSplit = 1;

    OP_CHECK_IF(SetSysWorkspace(context, tiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetSysWorkspace error"),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(1);
    SetTilingKeyByDtype(context, tiling);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PReluGradReduce)
    .Tiling(PReluGradReduceTilingFunc);
IMPL_OP_OPTILING(PReluGradReducePartial)
    .Tiling(PReluGradReducePartialTilingFunc);
IMPL_OP_OPTILING(PReluGradReduceFinal)
    .Tiling(PReluGradReduceFinalTilingFunc);
} // namespace optiling
