/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#ifndef P_RELU_GRAD_REDUCE_H_
#define P_RELU_GRAD_REDUCE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "p_relu_grad_reduce_tiling_data.h"

namespace NsPReluGradReduce {

constexpr uint32_t MAX_RANK = 4;
constexpr uint32_t DA_ALIGN = 16;
constexpr uint32_t FP32_DA_ALIGN = 8;
constexpr uint32_t V19_UB_REDUCE_ELEMS = 16384; 
constexpr uint32_t V19_REDUCE_TMP_ELEMS = 16384;
constexpr uint32_t REDUCE_MODE_ALL = 0;
constexpr uint32_t REDUCE_MODE_NCHW_CHANNEL = 1;
constexpr uint32_t REDUCE_MODE_ND_WEIGHT_SHAPE = 2;
constexpr uint32_t REDUCE_MODE_NC1HWC0 = 3;
constexpr uint32_t DTYPE_MODE_FP16 = 0;
constexpr uint32_t DTYPE_MODE_FP32 = 1;
constexpr uint32_t DTYPE_MODE_BF16 = 2;

class PReluGradReduce {
public:
    __aicore__ inline PReluGradReduce() {}

    __aicore__ inline void Init(GM_ADDR updates, GM_ADDR da, GM_ADDR workspace,
        const PReluGradReduceTilingData* tilingData)
    {
        reduceMode_ = tilingData->reduceMode;
        dtypeMode_ = tilingData->dtypeMode;
        totalLength_ = tilingData->totalLength;
        outputLength_ = tilingData->outputLength;
        daPaddedLength_ = tilingData->daPaddedLength == 0 ? DA_ALIGN : tilingData->daPaddedLength;
        blockDim_ = tilingData->blockDim == 0 ? 1 : tilingData->blockDim;
        coreNum_ = tilingData->coreNum == 0 ? 1 : tilingData->coreNum;
        reduceSplit_ = tilingData->reduceSplit == 0 ? 1 : tilingData->reduceSplit;
        sysWorkspaceSize_ = tilingData->sysWorkspaceSize;
        updatesRank_ = tilingData->updatesRank;
        weightsRank_ = tilingData->weightsRank;
        n_ = tilingData->n == 0 ? 1 : tilingData->n;
        c_ = tilingData->c == 0 ? 1 : tilingData->c;
        h_ = tilingData->h == 0 ? 1 : tilingData->h;
        w_ = tilingData->w == 0 ? 1 : tilingData->w;
        c1_ = tilingData->c1 == 0 ? 1 : tilingData->c1;
        c0_ = tilingData->c0 == 0 ? 1 : tilingData->c0;
        updatesShape_[0] = tilingData->updatesShape0;
        updatesShape_[1] = tilingData->updatesShape1;
        updatesShape_[2] = tilingData->updatesShape2;
        updatesShape_[3] = tilingData->updatesShape3;
        weightsShape_[0] = tilingData->weightsShape0;
        weightsShape_[1] = tilingData->weightsShape1;
        weightsShape_[2] = tilingData->weightsShape2;

        updatesHalfGm_.SetGlobalBuffer((__gm__ half*)updates, totalLength_);
        daHalfGm_.SetGlobalBuffer((__gm__ half*)da, outputLength_ == 0 ? 1 : outputLength_);
        updatesFloatGm_.SetGlobalBuffer((__gm__ float*)updates, totalLength_);
        daFloatGm_.SetGlobalBuffer((__gm__ float*)da, outputLength_ == 0 ? 1 : outputLength_);
        updatesBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t*)updates, totalLength_);
        daBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t*)da, outputLength_ == 0 ? 1 : outputLength_);

        // REDUCE_MODE_ALL 真正的多核 map-reduce 需要的两块 workspace（紧跟系统库 workspace 之后）：
        //   [sysWorkspaceSize_, +coreNum_*32B)        SyncAll 跨核同步区（用前清零）
        //   [..., +coreNum_*4B，32B对齐)               各核局部和（每核一个 float 槛位）
        // 这两块大小与 host 侧 GetWorkspaceSize() 的计算必须保持一致。
        if (reduceMode_ == REDUCE_MODE_ALL) {
            __gm__ uint8_t* wsBase = (__gm__ uint8_t*)workspace;
            size_t syncBytes = static_cast<size_t>(coreNum_) * 32;
            syncGm_.SetGlobalBuffer((__gm__ int32_t*)(wsBase + sysWorkspaceSize_), coreNum_ * 8);
            partialSumGm_.SetGlobalBuffer((__gm__ float*)(wsBase + sysWorkspaceSize_ + syncBytes), coreNum_);
            pipe_.InitBuffer(syncUbBuf_, coreNum_ * 32);
        } else if (reduceMode_ == REDUCE_MODE_NCHW_CHANNEL && reduceSplit_ > 1) {
            __gm__ uint8_t* wsBase = (__gm__ uint8_t*)workspace;
            size_t syncBytes = static_cast<size_t>(blockDim_) * 32;
            syncGm_.SetGlobalBuffer((__gm__ int32_t*)(wsBase + sysWorkspaceSize_), blockDim_ * 8);
            partialSumGm_.SetGlobalBuffer((__gm__ float*)(wsBase + sysWorkspaceSize_ + syncBytes),
                outputLength_ * reduceSplit_);
            pipe_.InitBuffer(syncUbBuf_, blockDim_ * 32);
        }

        pipe_.InitBuffer(outBuf_, 256 * sizeof(float));
        pipe_.InitBuffer(inBuf_, V19_UB_REDUCE_ELEMS * sizeof(float));
        pipe_.InitBuffer(reduceDstBuf_, 256 * sizeof(float));
        pipe_.InitBuffer(reduceTmpBuf_, V19_REDUCE_TMP_ELEMS * sizeof(float));
    }


    __aicore__ inline void Process()
    {
        if (reduceMode_ == REDUCE_MODE_ALL) {
            ProcessSharedMultiCore();
            return;
        }

        if (reduceMode_ == REDUCE_MODE_NC1HWC0) {
            ProcessNc1hwc0_Fast();
            return;
        }

        if (reduceMode_ == REDUCE_MODE_NCHW_CHANNEL && reduceSplit_ > 1) {
            ProcessChannelReduceSplit();
            return;
        }

        if (reduceMode_ == REDUCE_MODE_ND_WEIGHT_SHAPE && IsNdTailShapeFastPath()) {
            ProcessNdWeightShapeTailFast();
            return;
        }

        if (dtypeMode_ == DTYPE_MODE_FP32) {
            ProcessOutputParallelFp32();
        } else if (dtypeMode_ == DTYPE_MODE_BF16) {
            ProcessOutputParallelBf16();
        } else {
            ProcessOutputParallelFp16();
        }
    }

private:
    // 🚀 核心修复：纯字节级内存拷贝，无视 LLVM TBAA 严格别名检查，永不被优化为 0
    __aicore__ inline float CastBf16ToFloat(bfloat16_t val) {
        uint16_t u16_val;
        uint8_t* p_val = (uint8_t*)(&val);
        uint8_t* p_u16 = (uint8_t*)(&u16_val);
        p_u16[0] = p_val[0]; p_u16[1] = p_val[1];

        uint32_t u32_val = (uint32_t)(u16_val) << 16;
        float res;
        uint8_t* p_res = (uint8_t*)(&res);
        uint8_t* p_u32 = (uint8_t*)(&u32_val);
        p_res[0] = p_u32[0]; p_res[1] = p_u32[1]; p_res[2] = p_u32[2]; p_res[3] = p_u32[3];
        return res;
    }

    __aicore__ inline bfloat16_t CastFloatToBf16(float val) {
        uint32_t u32_val;
        uint8_t* p_val = (uint8_t*)(&val);
        uint8_t* p_u32 = (uint8_t*)(&u32_val);
        p_u32[0] = p_val[0]; p_u32[1] = p_val[1]; p_u32[2] = p_val[2]; p_u32[3] = p_val[3];

        uint32_t lsb = (u32_val >> 16) & 1;     // RNE 补偿逻辑，找回那丢失的 4096 (1 bit)
        u32_val += 0x7fff + lsb;
        uint16_t u16_val = (uint16_t)(u32_val >> 16);

        bfloat16_t res;
        uint8_t* p_res = (uint8_t*)(&res);
        uint8_t* p_u16 = (uint8_t*)(&u16_val);
        p_res[0] = p_u16[0]; p_res[1] = p_u16[1];
        return res;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // REDUCE_MODE_ALL —— 真正的多核 map-reduce（单次 kernel 启动内用 SyncAll 同步）
    //
    // 1) 每核把整段 SyncAll 同步区清零（官方文档推荐的 kernel 侧初始化方式：
    //    每个核都把全部同步区清零，多核写同一个值 0 互不冲突，不需要额外的跨核顺序保证）
    // 2) 每核对自己负责的 [start, end) 区间求局部和，写到 workspace 里自己的槛位（无竞争）
    // 3) AscendC::SyncAll() 跨核同步，确保所有核的清零 + 局部和写入都已落地可见
    // 4) 仅 core0 把 blockDim_ 个局部和加起来，写最终结果
    // ─────────────────────────────────────────────────────────────────────────
    __aicore__ inline void ProcessSharedMultiCore()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t cn = blockDim_ == 0 ? 1 : blockDim_;
        if (blockIdx >= cn) return;

        // 数据量很小、tiling 只下发了 1 个核时，直接单核算，不需要任何跨核同步
        if (cn <= 1) {
            if (blockIdx != 0) return;
            float sum = 0.0F;
            if (dtypeMode_ == DTYPE_MODE_FP32) {
                sum = SumRangeFp32(0, totalLength_);
            } else if (dtypeMode_ == DTYPE_MODE_BF16) {
                sum = SumRangeBf16(0, totalLength_);
            } else {
                sum = SumRangeFp16(0, totalLength_);
            }
            WriteSharedOutput(sum);
            return;
        }

        // 1) 均分 totalLength_，按对齐边界切块，最后一核兜底吃掉尾部
        uint32_t align = (dtypeMode_ == DTYPE_MODE_FP32) ? FP32_DA_ALIGN : DA_ALIGN;
        uint32_t base = totalLength_ / cn;
        base = (base / align) * align;
        if (base == 0) base = align;
        uint32_t start = blockIdx * base;
        uint32_t end = (blockIdx == cn - 1) ? totalLength_ : start + base;
        if (end > totalLength_) end = totalLength_;
        if (start > end) start = end;

        float partialSum = 0.0F;
        if (start < end) {
            if (dtypeMode_ == DTYPE_MODE_FP32) {
                partialSum = SumRangeFp32(start, end);
            } else if (dtypeMode_ == DTYPE_MODE_BF16) {
                partialSum = SumRangeBf16(start, end);
            } else {
                partialSum = SumRangeFp16(start, end);
            }
        }

        // 写到自己的槛位，各核位置不同，无竞争
        partialSumGm_.SetValue(blockIdx, partialSum);
        AscendC::PipeBarrier<PIPE_MTE3>();

        // 2) 软同步：workspace 由 acl 调用侧清零，这里只等待各核完成局部和写入
        AscendC::LocalTensor<int32_t> ubSync = syncUbBuf_.Get<int32_t>();
        AscendC::SyncAll<true>(syncGm_, ubSync, static_cast<int32_t>(cn));

        // 3) 仅 core0 汇总并写最终结果
        if (blockIdx == 0) {
            float total = 0.0F;
            for (uint32_t i = 0; i < cn; ++i) {
                total += partialSumGm_.GetValue(i);
            }
            WriteSharedOutput(total);
        }
    }

    // =========================================================================
    // 🚀 V17 对折算法 + 绝对对齐修复 + 纯标量安全写回
    // =========================================================================
    __aicore__ inline void ProcessNc1hwc0_Fast() {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        if (blockNum == 0) blockNum = 1;

        uint32_t c1_per_core = (c1_ + blockNum - 1) / blockNum;
        uint32_t start_c1 = blockIdx * c1_per_core;
        uint32_t end_c1 = start_c1 + c1_per_core;
        if (start_c1 >= c1_) return;
        if (end_c1 > c1_) end_c1 = c1_;

        uint32_t hwTotal = h_ * w_;
        uint32_t tileElements = V19_UB_REDUCE_ELEMS;

        for (uint32_t c1Idx = start_c1; c1Idx < end_c1; ++c1Idx) {
            float sum_c0[256];
            for (uint32_t j = 0; j < 256; ++j) sum_c0[j] = 0.0f;

            for (uint32_t nIdx = 0; nIdx < n_; ++nIdx) {
                uint32_t baseOffset = (nIdx * c1_ + c1Idx) * hwTotal * c0_;
                uint32_t elementsLeft = hwTotal * c0_;
                uint32_t offset = 0;

                while (elementsLeft > 0) {
                    uint32_t currentChunk = (elementsLeft < tileElements) ? elementsLeft : tileElements;
                    
                    uint32_t alignElem = (dtypeMode_ == DTYPE_MODE_FP32) ? FP32_DA_ALIGN : DA_ALIGN;
                    uint32_t copyChunk = ((currentChunk + alignElem - 1) / alignElem) * alignElem;
                    uint32_t castChunk = ((copyChunk + 63) / 64) * 64;

                    AscendC::LocalTensor<float> calcF32;
                    
                    if (dtypeMode_ == DTYPE_MODE_FP32) {
                        calcF32 = inBuf_.Get<float>();
                        AscendC::DataCopy(calcF32, updatesFloatGm_[baseOffset + offset], copyChunk);
                        AscendC::PipeBarrier<PIPE_ALL>(); 
                        for (uint32_t m = currentChunk; m < castChunk; ++m) calcF32.SetValue(m, 0.0f);
                        AscendC::PipeBarrier<PIPE_ALL>(); 
                    } else if (dtypeMode_ == DTYPE_MODE_BF16) {
                        calcF32 = reduceTmpBuf_.Get<float>();
                        AscendC::LocalTensor<bfloat16_t> inLocal = inBuf_.Get<bfloat16_t>();
                        AscendC::DataCopy(inLocal, updatesBf16Gm_[baseOffset + offset], copyChunk);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        
                        // 字节级安全纯量 Cast
                        for (uint32_t m = 0; m < currentChunk; ++m) {
                            calcF32.SetValue(m, CastBf16ToFloat(inLocal.GetValue(m)));
                        }
                        for (uint32_t m = currentChunk; m < castChunk; ++m) {
                            calcF32.SetValue(m, 0.0f);
                        }
                        AscendC::PipeBarrier<PIPE_ALL>();
                    } else {
                        calcF32 = reduceTmpBuf_.Get<float>();
                        AscendC::LocalTensor<half> inLocal = inBuf_.Get<half>();
                        AscendC::DataCopy(inLocal, updatesHalfGm_[baseOffset + offset], copyChunk);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        for (uint32_t m = currentChunk; m < castChunk; ++m) inLocal.SetValue(m, static_cast<half>(0.0f));
                        AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::Cast(calcF32, inLocal, AscendC::RoundMode::CAST_NONE, castChunk); 
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }

                    // 完美复刻 V17 的折叠算法
                    if (currentChunk == tileElements) {
                        uint32_t foldLen = tileElements / 2;
                        while (foldLen >= 64) {
                            AscendC::Add(calcF32, calcF32, calcF32[foldLen], foldLen);
                            AscendC::PipeBarrier<PIPE_ALL>();
                            foldLen /= 2;
                        }
                        for (uint32_t j = 0; j < 64; ++j) {
                            sum_c0[j % c0_] += calcF32.GetValue(j);
                        }
                    } else {
                        uint32_t tailChunks = currentChunk / 64;
                        if (tailChunks > 0) {
                            for (uint32_t j = 1; j < tailChunks; ++j) {
                                AscendC::Add(calcF32, calcF32, calcF32[j * 64], 64);
                                AscendC::PipeBarrier<PIPE_ALL>();
                            }
                            for (uint32_t j = 0; j < 64; ++j) {
                                sum_c0[j % c0_] += calcF32.GetValue(j);
                            }
                        }
                        for (uint32_t j = tailChunks * 64; j < currentChunk; ++j) {
                            sum_c0[j % c0_] += calcF32.GetValue(j);
                        }
                    }

                    elementsLeft -= currentChunk;
                    offset += currentChunk;
                }
            }

            uint32_t outBase = c1Idx * c0_;
            uint32_t alignOut = (dtypeMode_ == DTYPE_MODE_FP32) ? FP32_DA_ALIGN : DA_ALIGN;
            uint32_t alignedC0 = ((c0_ + alignOut - 1) / alignOut) * alignOut;
            
            if (dtypeMode_ == DTYPE_MODE_FP32) {
                AscendC::LocalTensor<float> outLocal = outBuf_.Get<float>();
                for (uint32_t j = 0; j < c0_; ++j) outLocal.SetValue(j, sum_c0[j]);
                for (uint32_t j = c0_; j < alignedC0; ++j) outLocal.SetValue(j, 0.0f);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(daFloatGm_[outBase], outLocal, alignedC0);
            } else if (dtypeMode_ == DTYPE_MODE_BF16) {
                AscendC::LocalTensor<bfloat16_t> outLocal = outBuf_.Get<bfloat16_t>();
                for (uint32_t j = 0; j < c0_; ++j) outLocal.SetValue(j, CastFloatToBf16(sum_c0[j]));
                for (uint32_t j = c0_; j < alignedC0; ++j) outLocal.SetValue(j, CastFloatToBf16(0.0f));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(daBf16Gm_[outBase], outLocal, alignedC0);
            } else {
                AscendC::LocalTensor<half> outLocal = outBuf_.Get<half>();
                for (uint32_t j = 0; j < c0_; ++j) outLocal.SetValue(j, static_cast<half>(sum_c0[j]));
                for (uint32_t j = c0_; j < alignedC0; ++j) outLocal.SetValue(j, static_cast<half>(0.0f));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(daHalfGm_[outBase], outLocal, alignedC0);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    // =========================================================================

    __aicore__ inline uint32_t RoundUpPowerOfTwo(uint32_t value)
    {
        uint32_t rounded = 1;
        while (rounded < value) {
            rounded <<= 1;
        }
        return rounded < 64 ? 64 : rounded;
    }

    __aicore__ inline float FoldSumF32(AscendC::LocalTensor<float> calcF32, uint32_t validLen, uint32_t reduceLen)
    {
        for (uint32_t m = validLen; m < reduceLen; ++m) {
            calcF32.SetValue(m, 0.0F);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        uint32_t foldLen = reduceLen / 2;
        while (foldLen >= 64) {
            AscendC::Add(calcF32, calcF32, calcF32[foldLen], foldLen);
            AscendC::PipeBarrier<PIPE_ALL>();
            foldLen /= 2;
        }

        float sum = 0.0F;
        for (uint32_t k = 0; k < 64; ++k) {
            sum += calcF32.GetValue(k);
        }
        return sum;
    }

    __aicore__ inline float SumRangeFp16(uint32_t start, uint32_t end)
    {
        float sum = 0.0F;
        uint32_t i = start;
        AscendC::LocalTensor<half> inLocal = inBuf_.Get<half>();
        AscendC::LocalTensor<float> calcF32 = reduceTmpBuf_.Get<float>();
        
        while (i < end && (i % DA_ALIGN != 0)) {
            sum += static_cast<float>(updatesHalfGm_.GetValue(i));
            ++i;
        }
        while (i + DA_ALIGN <= end) {
            uint32_t copyLen = end - i;
            if (copyLen > V19_UB_REDUCE_ELEMS) copyLen = V19_UB_REDUCE_ELEMS;
            copyLen = (copyLen / DA_ALIGN) * DA_ALIGN;
            if (copyLen == 0) break;
            
            AscendC::DataCopy(inLocal, updatesHalfGm_[i], copyLen);
            AscendC::PipeBarrier<PIPE_ALL>();
            uint32_t reduceLen = RoundUpPowerOfTwo(copyLen);
            for (uint32_t k = copyLen; k < reduceLen; ++k) {
                inLocal.SetValue(k, static_cast<half>(0.0F));
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(calcF32, inLocal, AscendC::RoundMode::CAST_NONE, reduceLen);
            AscendC::PipeBarrier<PIPE_ALL>();
            sum += FoldSumF32(calcF32, copyLen, reduceLen);
            i += copyLen;
        }
        while (i < end) {
            sum += static_cast<float>(updatesHalfGm_.GetValue(i));
            ++i;
        }
        return sum;
    }

    __aicore__ inline float SumRangeFp32(uint32_t start, uint32_t end)
    {
        float sum = 0.0F;
        uint32_t i = start;
        AscendC::LocalTensor<float> inLocal = inBuf_.Get<float>();
        
        while (i < end && (i % 8 != 0)) {
            sum += updatesFloatGm_.GetValue(i);
            ++i;
        }
        while (i + 8 <= end) {
            uint32_t copyLen = end - i;
            if (copyLen > V19_UB_REDUCE_ELEMS) copyLen = V19_UB_REDUCE_ELEMS;
            copyLen = (copyLen / 8) * 8;
            if (copyLen == 0) break;
            
            AscendC::DataCopy(inLocal, updatesFloatGm_[i], copyLen);
            AscendC::PipeBarrier<PIPE_ALL>();
            uint32_t reduceLen = RoundUpPowerOfTwo(copyLen);
            sum += FoldSumF32(inLocal, copyLen, reduceLen);
            i += copyLen;
        }
        while (i < end) {
            sum += updatesFloatGm_.GetValue(i);
            ++i;
        }
        return sum;
    }

    __aicore__ inline float SumRangeBf16(uint32_t start, uint32_t end)
    {
        float sum = 0.0F;
        uint32_t i = start;
        AscendC::LocalTensor<bfloat16_t> inLocal = inBuf_.Get<bfloat16_t>();
        AscendC::LocalTensor<float> calcF32 = reduceTmpBuf_.Get<float>();
        
        while (i < end && (i % DA_ALIGN != 0)) {
            sum += CastBf16ToFloat(updatesBf16Gm_.GetValue(i));
            ++i;
        }
        while (i + DA_ALIGN <= end) {
            uint32_t copyLen = end - i;
            if (copyLen > V19_UB_REDUCE_ELEMS) copyLen = V19_UB_REDUCE_ELEMS;
            copyLen = (copyLen / DA_ALIGN) * DA_ALIGN;
            if (copyLen == 0) break;

            AscendC::DataCopy(inLocal, updatesBf16Gm_[i], copyLen);
            AscendC::PipeBarrier<PIPE_ALL>();
            uint32_t reduceLen = RoundUpPowerOfTwo(copyLen);
            for (uint32_t k = 0; k < copyLen; ++k) {
                calcF32.SetValue(k, CastBf16ToFloat(inLocal.GetValue(k)));
            }
            for (uint32_t k = copyLen; k < reduceLen; ++k) {
                calcF32.SetValue(k, 0.0F);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            sum += FoldSumF32(calcF32, copyLen, reduceLen);
            i += copyLen;
        }
        while (i < end) {
            sum += CastBf16ToFloat(updatesBf16Gm_.GetValue(i));
            ++i;
        }
        return sum;
    }

    __aicore__ inline void WriteSharedOutput(float sum)
    {
        if (dtypeMode_ == DTYPE_MODE_FP32) {
            daFloatGm_.SetValue(0, sum);
        } else if (dtypeMode_ == DTYPE_MODE_FP16) {
            daHalfGm_.SetValue(0, static_cast<half>(sum));
        } else {
            AscendC::LocalTensor<float> accF32 = reduceDstBuf_.Get<float>();
            accF32.SetValue(0, sum);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::LocalTensor<bfloat16_t> outLocal = outBuf_.Get<bfloat16_t>();
            outLocal.SetValue(0, CastFloatToBf16(accF32.GetValue(0)));
            AscendC::PipeBarrier<PIPE_ALL>();
            daBf16Gm_.SetValue(0, outLocal.GetValue(0)); 
        }
    }

    __aicore__ inline float SumRangeByDtype(uint32_t start, uint32_t end)
    {
        if (dtypeMode_ == DTYPE_MODE_FP32) return SumRangeFp32(start, end);
        if (dtypeMode_ == DTYPE_MODE_BF16) return SumRangeBf16(start, end);
        return SumRangeFp16(start, end);
    }

    __aicore__ inline void WriteChannelOutput(uint32_t channel, float sum)
    {
        if (dtypeMode_ == DTYPE_MODE_FP32) {
            daFloatGm_.SetValue(channel, sum);
        } else if (dtypeMode_ == DTYPE_MODE_FP16) {
            daHalfGm_.SetValue(channel, static_cast<half>(sum));
        } else {
            daBf16Gm_.SetValue(channel, CastFloatToBf16(sum));
        }
    }

    __aicore__ inline float ComputeChannelSlice(uint32_t channel, uint32_t elemStart, uint32_t elemEnd)
    {
        float sum = 0.0F;
        uint32_t elem = elemStart;
        while (elem < elemEnd) {
            uint32_t row = elem / w_;
            uint32_t wStart = elem - row * w_;
            uint32_t nIdx = row / h_;
            uint32_t hIdx = row - nIdx * h_;
            uint32_t rowEnd = (row + 1) * w_;
            uint32_t spanEnd = elemEnd < rowEnd ? elemEnd : rowEnd;
            uint32_t offset = ((nIdx * c_ + channel) * h_ + hIdx) * w_ + wStart;
            sum += SumRangeByDtype(offset, offset + (spanEnd - elem));
            elem = spanEnd;
        }
        return sum;
    }

    __aicore__ inline void ProcessChannelReduceSplit()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t cn = blockDim_ == 0 ? 1 : blockDim_;
        if (blockIdx >= cn) return;

        uint32_t reduceSplit = reduceSplit_ == 0 ? 1 : reduceSplit_;
        uint32_t channel = blockIdx / reduceSplit;
        uint32_t reduceSlice = blockIdx - channel * reduceSplit;
        if (channel >= outputLength_) return;
        uint32_t reduceElems = n_ * h_ * w_;
        uint32_t elemsPerSlice = (reduceElems + reduceSplit - 1) / reduceSplit;
        uint32_t elemStart = reduceSlice * elemsPerSlice;
        uint32_t elemEnd = elemStart + elemsPerSlice;
        if (elemStart > reduceElems) elemStart = reduceElems;
        if (elemEnd > reduceElems) elemEnd = reduceElems;

        float partial = 0.0F;
        if (elemStart < elemEnd) {
            partial = ComputeChannelSlice(channel, elemStart, elemEnd);
        }
        partialSumGm_.SetValue(channel * reduceSplit + reduceSlice, partial);
        AscendC::PipeBarrier<PIPE_MTE3>();
        AscendC::LocalTensor<int32_t> ubSync = syncUbBuf_.Get<int32_t>();
        AscendC::SyncAll<true>(syncGm_, ubSync, static_cast<int32_t>(cn));

        if (reduceSlice == 0) {
            float total = 0.0F;
            for (uint32_t i = 0; i < reduceSplit; ++i) {
                total += partialSumGm_.GetValue(channel * reduceSplit + i);
            }
            WriteChannelOutput(channel, total);
        }
    }

    __aicore__ inline void ProcessOutputParallelFp16()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockDim_) return;

        AscendC::LocalTensor<half> outLocal = outBuf_.Get<half>();
        uint32_t chunkCount = daPaddedLength_ / DA_ALIGN;
        for (uint32_t chunkIdx = blockIdx; chunkIdx < chunkCount; chunkIdx += blockDim_) {
            uint32_t chunkStart = chunkIdx * DA_ALIGN;
            for (uint32_t lane = 0; lane < DA_ALIGN; ++lane) {
                uint32_t outIndex = chunkStart + lane;
                half value = outIndex < outputLength_ ? static_cast<half>(ComputeOutput(outIndex)) :
                    static_cast<half>(0.0F);
                outLocal.SetValue(lane, value);
            }
            if (chunkStart + DA_ALIGN <= outputLength_) {
                AscendC::DataCopy(daHalfGm_[chunkStart], outLocal, DA_ALIGN);
            } else {
                for (uint32_t lane = 0; lane < DA_ALIGN; ++lane) {
                    uint32_t outIndex = chunkStart + lane;
                    if (outIndex < outputLength_) {
                        daHalfGm_.SetValue(outIndex, outLocal.GetValue(lane));
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessOutputParallelFp32()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockDim_) return;

        AscendC::LocalTensor<float> outLocal = outBuf_.Get<float>();
        uint32_t outputPaddedLength = ((outputLength_ + FP32_DA_ALIGN - 1) / FP32_DA_ALIGN) * FP32_DA_ALIGN;
        uint32_t chunkCount = outputPaddedLength / FP32_DA_ALIGN;
        for (uint32_t chunkIdx = blockIdx; chunkIdx < chunkCount; chunkIdx += blockDim_) {
            uint32_t chunkStart = chunkIdx * FP32_DA_ALIGN;
            for (uint32_t lane = 0; lane < FP32_DA_ALIGN; ++lane) {
                uint32_t outIndex = chunkStart + lane;
                float value = outIndex < outputLength_ ? ComputeOutput(outIndex) : 0.0F;
                outLocal.SetValue(lane, value);
            }
            if (chunkStart + FP32_DA_ALIGN <= outputLength_) {
                AscendC::DataCopy(daFloatGm_[chunkStart], outLocal, FP32_DA_ALIGN);
            } else {
                for (uint32_t lane = 0; lane < FP32_DA_ALIGN; ++lane) {
                    uint32_t outIndex = chunkStart + lane;
                    if (outIndex < outputLength_) {
                        daFloatGm_.SetValue(outIndex, outLocal.GetValue(lane));
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessOutputParallelBf16()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockDim_) return;

        AscendC::LocalTensor<bfloat16_t> outLocal = outBuf_.Get<bfloat16_t>();
        AscendC::LocalTensor<float> accF32 = reduceDstBuf_.Get<float>();
        
        uint32_t chunkCount = daPaddedLength_ / DA_ALIGN;
        for (uint32_t chunkIdx = blockIdx; chunkIdx < chunkCount; chunkIdx += blockDim_) {
            uint32_t chunkStart = chunkIdx * DA_ALIGN;
            for (uint32_t lane = 0; lane < DA_ALIGN; ++lane) {
                uint32_t outIndex = chunkStart + lane;
                float value = outIndex < outputLength_ ? ComputeOutput(outIndex) : 0.0F;
                accF32.SetValue(lane, value);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t lane = 0; lane < DA_ALIGN; ++lane) {
                outLocal.SetValue(lane, CastFloatToBf16(accF32.GetValue(lane)));
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            
            if (chunkStart + DA_ALIGN <= outputLength_) {
                AscendC::DataCopy(daBf16Gm_[chunkStart], outLocal, DA_ALIGN);
            } else {
                for (uint32_t lane = 0; lane < DA_ALIGN; ++lane) {
                    uint32_t outIndex = chunkStart + lane;
                    if (outIndex < outputLength_) {
                        daBf16Gm_.SetValue(outIndex, outLocal.GetValue(lane));
                    }
                }
            }
        }
    }

    __aicore__ inline bool IsNdTailShapeFastPath()
    {
        if (updatesRank_ == 3 && weightsRank_ == 2) {
            return weightsShape_[0] == updatesShape_[1] && weightsShape_[1] == updatesShape_[2];
        }
        if (updatesRank_ == 4 && weightsRank_ == 3) {
            return weightsShape_[0] == updatesShape_[1] && weightsShape_[1] == updatesShape_[2] &&
                weightsShape_[2] == updatesShape_[3];
        }
        return false;
    }

    __aicore__ inline void InitF32Tile(AscendC::LocalTensor<float> tensor, uint32_t len)
    {
        for (uint32_t i = 0; i < len; ++i) {
            tensor.SetValue(i, 0.0F);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadNdTailTileToF32(uint32_t gmOffset, uint32_t tileLen, uint32_t calcLen,
        AscendC::LocalTensor<float> tileF32)
    {
        uint32_t copyLen = 0;
        if (dtypeMode_ == DTYPE_MODE_FP32) {
            copyLen = (tileLen / FP32_DA_ALIGN) * FP32_DA_ALIGN;
            if (copyLen > 0) {
                AscendC::DataCopy(tileF32, updatesFloatGm_[gmOffset], copyLen);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            for (uint32_t i = copyLen; i < tileLen; ++i) {
                tileF32.SetValue(i, updatesFloatGm_.GetValue(gmOffset + i));
            }
        } else if (dtypeMode_ == DTYPE_MODE_FP16) {
            AscendC::LocalTensor<half> inLocal = inBuf_.Get<half>();
            copyLen = (tileLen / DA_ALIGN) * DA_ALIGN;
            if (copyLen > 0) {
                AscendC::DataCopy(inLocal, updatesHalfGm_[gmOffset], copyLen);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Cast(tileF32, inLocal, AscendC::RoundMode::CAST_NONE, copyLen);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            for (uint32_t i = copyLen; i < tileLen; ++i) {
                tileF32.SetValue(i, static_cast<float>(updatesHalfGm_.GetValue(gmOffset + i)));
            }
        } else {
            AscendC::LocalTensor<bfloat16_t> inLocal = inBuf_.Get<bfloat16_t>();
            copyLen = (tileLen / DA_ALIGN) * DA_ALIGN;
            if (copyLen > 0) {
                AscendC::DataCopy(inLocal, updatesBf16Gm_[gmOffset], copyLen);
                AscendC::PipeBarrier<PIPE_ALL>();
                for (uint32_t i = 0; i < copyLen; ++i) {
                    tileF32.SetValue(i, CastBf16ToFloat(inLocal.GetValue(i)));
                }
            }
            for (uint32_t i = copyLen; i < tileLen; ++i) {
                tileF32.SetValue(i, CastBf16ToFloat(updatesBf16Gm_.GetValue(gmOffset + i)));
            }
        }
        for (uint32_t i = tileLen; i < calcLen; ++i) {
            tileF32.SetValue(i, 0.0F);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WriteNdTailTile(uint32_t tileStart, uint32_t tileLen, uint32_t calcLen,
        AscendC::LocalTensor<float> accF32)
    {
        if (dtypeMode_ == DTYPE_MODE_FP32) {
            if (tileLen == calcLen && (tileLen % FP32_DA_ALIGN) == 0) {
                AscendC::DataCopy(daFloatGm_[tileStart], accF32, tileLen);
            } else {
                for (uint32_t i = 0; i < tileLen; ++i) {
                    daFloatGm_.SetValue(tileStart + i, accF32.GetValue(i));
                }
            }
        } else if (dtypeMode_ == DTYPE_MODE_FP16) {
            AscendC::LocalTensor<half> outLocal = outBuf_.Get<half>();
            AscendC::Cast(outLocal, accF32, AscendC::RoundMode::CAST_NONE, calcLen);
            AscendC::PipeBarrier<PIPE_ALL>();
            if (tileLen == calcLen && (tileLen % DA_ALIGN) == 0) {
                AscendC::DataCopy(daHalfGm_[tileStart], outLocal, tileLen);
            } else {
                for (uint32_t i = 0; i < tileLen; ++i) {
                    daHalfGm_.SetValue(tileStart + i, outLocal.GetValue(i));
                }
            }
        } else {
            AscendC::LocalTensor<bfloat16_t> outLocal = outBuf_.Get<bfloat16_t>();
            for (uint32_t i = 0; i < calcLen; ++i) {
                outLocal.SetValue(i, CastFloatToBf16(accF32.GetValue(i)));
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            if (tileLen == calcLen && (tileLen % DA_ALIGN) == 0) {
                AscendC::DataCopy(daBf16Gm_[tileStart], outLocal, tileLen);
            } else {
                for (uint32_t i = 0; i < tileLen; ++i) {
                    daBf16Gm_.SetValue(tileStart + i, outLocal.GetValue(i));
                }
            }
        }
    }

    __aicore__ inline void ProcessNdWeightShapeTailFast()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= blockDim_) return;

        constexpr uint32_t TILE_ELEMS = 256;
        AscendC::LocalTensor<float> accF32 = reduceDstBuf_.Get<float>();
        AscendC::LocalTensor<float> tileF32 = reduceTmpBuf_.Get<float>();
        uint32_t tileCount = (outputLength_ + TILE_ELEMS - 1) / TILE_ELEMS;

        for (uint32_t tileIdx = blockIdx; tileIdx < tileCount; tileIdx += blockDim_) {
            uint32_t tileStart = tileIdx * TILE_ELEMS;
            uint32_t tileLen = outputLength_ - tileStart;
            if (tileLen > TILE_ELEMS) tileLen = TILE_ELEMS;
            uint32_t calcLen = RoundUpPowerOfTwo(tileLen);

            InitF32Tile(accF32, calcLen);
            for (uint32_t nIdx = 0; nIdx < updatesShape_[0]; ++nIdx) {
                uint32_t gmOffset = nIdx * outputLength_ + tileStart;
                LoadNdTailTileToF32(gmOffset, tileLen, calcLen, tileF32);
                AscendC::Add(accF32, accF32, tileF32, calcLen);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            WriteNdTailTile(tileStart, tileLen, calcLen, accF32);
        }
    }
// 👇 1. 在这里新增 SumContiguous 函数
    // 🚀 新增：专门用于连续内存块极速求和的引擎
    __aicore__ inline float SumContiguous(uint32_t start, uint32_t length)
    {
        if (length == 0) return 0.0F;
        float sum = 0.0F;
        uint32_t i = start;
        uint32_t end = start + length;

        if (dtypeMode_ == DTYPE_MODE_FP32) {
            AscendC::LocalTensor<float> inLocal = inBuf_.Get<float>();
            while (i < end && (i % FP32_DA_ALIGN != 0)) {
                sum += updatesFloatGm_.GetValue(i);
                ++i;
            }
            while (i + FP32_DA_ALIGN <= end) {
                uint32_t copyLen = end - i;
                if (copyLen > V19_UB_REDUCE_ELEMS) copyLen = V19_UB_REDUCE_ELEMS;
                copyLen = (copyLen / FP32_DA_ALIGN) * FP32_DA_ALIGN;
                AscendC::DataCopy(inLocal, updatesFloatGm_[i], copyLen);
                AscendC::PipeBarrier<PIPE_ALL>(); 
                for (uint32_t k = 0; k < copyLen; ++k) sum += inLocal.GetValue(k);
                i += copyLen;
            }
            while (i < end) {
                sum += updatesFloatGm_.GetValue(i);
                ++i;
            }
        } else if (dtypeMode_ == DTYPE_MODE_BF16) {
            AscendC::LocalTensor<bfloat16_t> inLocal = inBuf_.Get<bfloat16_t>();
            while (i < end && (i % DA_ALIGN != 0)) {
                sum += CastBf16ToFloat(updatesBf16Gm_.GetValue(i));
                ++i;
            }
            while (i + DA_ALIGN <= end) {
                uint32_t copyLen = end - i;
                if (copyLen > V19_UB_REDUCE_ELEMS) copyLen = V19_UB_REDUCE_ELEMS;
                copyLen = (copyLen / DA_ALIGN) * DA_ALIGN;
                AscendC::DataCopy(inLocal, updatesBf16Gm_[i], copyLen);
                AscendC::PipeBarrier<PIPE_ALL>(); 
                for (uint32_t k = 0; k < copyLen; ++k) sum += CastBf16ToFloat(inLocal.GetValue(k));
                i += copyLen;
            }
            while (i < end) {
                sum += CastBf16ToFloat(updatesBf16Gm_.GetValue(i));
                ++i;
            }
        } else {
            AscendC::LocalTensor<half> inLocal = inBuf_.Get<half>();
            while (i < end && (i % DA_ALIGN != 0)) {
                sum += static_cast<float>(updatesHalfGm_.GetValue(i));
                ++i;
            }
            while (i + DA_ALIGN <= end) {
                uint32_t copyLen = end - i;
                if (copyLen > V19_UB_REDUCE_ELEMS) copyLen = V19_UB_REDUCE_ELEMS;
                copyLen = (copyLen / DA_ALIGN) * DA_ALIGN;
                AscendC::DataCopy(inLocal, updatesHalfGm_[i], copyLen);
                AscendC::PipeBarrier<PIPE_ALL>(); 
                for (uint32_t k = 0; k < copyLen; ++k) sum += static_cast<float>(inLocal.GetValue(k));
                i += copyLen;
            }
            while (i < end) {
                sum += static_cast<float>(updatesHalfGm_.GetValue(i));
                ++i;
            }
        }
        return sum;
    }

    __aicore__ inline float ReadUpdate(uint32_t index)
    {
        if (dtypeMode_ == DTYPE_MODE_FP32) {
            return updatesFloatGm_.GetValue(index);
        } else if (dtypeMode_ == DTYPE_MODE_FP16) {
            return static_cast<float>(updatesHalfGm_.GetValue(index));
        } else {
            return CastBf16ToFloat(updatesBf16Gm_.GetValue(index));
        }
    }

    __aicore__ inline float ComputeOutput(uint32_t outIndex)
    {
        if (reduceMode_ == REDUCE_MODE_NCHW_CHANNEL) return ComputeChannel(outIndex);
        if (reduceMode_ == REDUCE_MODE_ND_WEIGHT_SHAPE) {
            if (updatesRank_ == 3 && weightsRank_ == 2) return ComputeNdRank3(outIndex);
            if (updatesRank_ == 4 && weightsRank_ == 3) return ComputeNdRank4(outIndex);
            return ComputeNdGeneric(outIndex);
        }
        return 0.0F;
    }

    __aicore__ inline float ComputeChannel(uint32_t channel)
    {
        float sum = 0.0F;
        for (uint32_t nIdx = 0; nIdx < n_; ++nIdx) {
            for (uint32_t hIdx = 0; hIdx < h_; ++hIdx) {
                uint32_t baseOffset = ((nIdx * c_ + channel) * h_ + hIdx) * w_;
                for (uint32_t wIdx = 0; wIdx < w_; ++wIdx) {
                    sum += ReadUpdate(baseOffset + wIdx);
                }
            }
        }
        return sum;
    }

    __aicore__ inline float ComputeNdRank3(uint32_t outIndex)
    {
        uint32_t outCoord0 = outIndex / weightsShape_[1];
        uint32_t outCoord1 = outIndex % weightsShape_[1];
        uint32_t dim1 = updatesShape_[1];
        uint32_t dim2 = updatesShape_[2];
        bool reduceDim1 = dim1 != weightsShape_[0];
        bool reduceDim2 = dim2 != weightsShape_[1];
        uint32_t dim1ReduceLength = reduceDim1 ? dim1 : 1;
        uint32_t dim2ReduceLength = reduceDim2 ? dim2 : 1;
        float sum = 0.0F;
        for (uint32_t nIdx = 0; nIdx < updatesShape_[0]; ++nIdx) {
            for (uint32_t coord1Reduce = 0; coord1Reduce < dim1ReduceLength; ++coord1Reduce) {
                uint32_t coord1 = reduceDim1 ? coord1Reduce : outCoord0;
                
                // 🚀 如果最内层是被 reduce 的，说明内存完全连续，启动块读取
                if (reduceDim2) {
                    uint32_t offset = (nIdx * dim1 + coord1) * dim2;
                    sum += SumContiguous(offset, dim2ReduceLength);
                } else {
                    // 如果是不连续的跳跃读取，保留你原汁原味的安全标量读取兜底
                    for (uint32_t coord2Reduce = 0; coord2Reduce < dim2ReduceLength; ++coord2Reduce) {
                        uint32_t coord2 = reduceDim2 ? coord2Reduce : outCoord1;
                        uint32_t offset = (nIdx * dim1 + coord1) * dim2 + coord2;
                        sum += ReadUpdate(offset);
                    }
                }
            }
        }
        return sum;
    }

    __aicore__ inline float ComputeNdRank4(uint32_t outIndex)
    {
        uint32_t outCoord0 = outIndex / (weightsShape_[1] * weightsShape_[2]);
        uint32_t restOut = outIndex % (weightsShape_[1] * weightsShape_[2]);
        uint32_t outCoord1 = restOut / weightsShape_[2];
        uint32_t outCoord2 = restOut % weightsShape_[2];
        uint32_t dim1 = updatesShape_[1];
        uint32_t dim2 = updatesShape_[2];
        uint32_t dim3 = updatesShape_[3];
        bool reduceDim1 = dim1 != weightsShape_[0];
        bool reduceDim2 = dim2 != weightsShape_[1];
        bool reduceDim3 = dim3 != weightsShape_[2];
        uint32_t dim1ReduceLength = reduceDim1 ? dim1 : 1;
        uint32_t dim2ReduceLength = reduceDim2 ? dim2 : 1;
        uint32_t dim3ReduceLength = reduceDim3 ? dim3 : 1;
        float sum = 0.0F;
        for (uint32_t nIdx = 0; nIdx < updatesShape_[0]; ++nIdx) {
            for (uint32_t coord1Reduce = 0; coord1Reduce < dim1ReduceLength; ++coord1Reduce) {
                uint32_t coord1 = reduceDim1 ? coord1Reduce : outCoord0;
                for (uint32_t coord2Reduce = 0; coord2Reduce < dim2ReduceLength; ++coord2Reduce) {
                    uint32_t coord2 = reduceDim2 ? coord2Reduce : outCoord1;
                    
                    // 🚀 判断最内层是否连续
                    if (reduceDim3) {
                        uint32_t offset = ((nIdx * dim1 + coord1) * dim2 + coord2) * dim3;
                        sum += SumContiguous(offset, dim3ReduceLength);
                    } else {
                        // 保留原有兜底
                        for (uint32_t coord3Reduce = 0; coord3Reduce < dim3ReduceLength; ++coord3Reduce) {
                            uint32_t coord3 = reduceDim3 ? coord3Reduce : outCoord2;
                            uint32_t offset = ((nIdx * dim1 + coord1) * dim2 + coord2) * dim3 + coord3;
                            sum += ReadUpdate(offset);
                        }
                    }
                }
            }
        }
        return sum;
    }
    __aicore__ inline float ComputeNdGeneric(uint32_t outIndex)
    {
        uint32_t outCoord0 = 0, outCoord1 = 0, outCoord2 = 0;
        DecodeOutputIndex(outIndex, outCoord0, outCoord1, outCoord2);
        float sum = 0.0F;
        for (uint32_t inIndex = 0; inIndex < totalLength_; ++inIndex) {
            uint32_t inCoord0 = 0, inCoord1 = 0, inCoord2 = 0, inCoord3 = 0;
            DecodeInputIndex(inIndex, inCoord0, inCoord1, inCoord2, inCoord3);
            if (InputMatchesOutput(inCoord1, inCoord2, inCoord3, outCoord0, outCoord1, outCoord2)) {
                sum += ReadUpdate(inIndex);
            }
        }
        return sum;
    }

    __aicore__ inline void DecodeOutputIndex(uint32_t index, uint32_t& coord0, uint32_t& coord1,
        uint32_t& coord2)
    {
        coord0 = 0; coord1 = 0; coord2 = 0;
        if (weightsRank_ == 1) {
            coord0 = index;
        } else if (weightsRank_ == 2) {
            coord0 = index / weightsShape_[1];
            coord1 = index % weightsShape_[1];
        } else {
            coord0 = index / (weightsShape_[1] * weightsShape_[2]);
            uint32_t rest = index % (weightsShape_[1] * weightsShape_[2]);
            coord1 = rest / weightsShape_[2];
            coord2 = rest % weightsShape_[2];
        }
    }

    __aicore__ inline void DecodeInputIndex(uint32_t index, uint32_t& coord0, uint32_t& coord1,
        uint32_t& coord2, uint32_t& coord3)
    {
        coord0 = 0; coord1 = 0; coord2 = 0; coord3 = 0;
        if (updatesRank_ == 3) {
            coord0 = index / (updatesShape_[1] * updatesShape_[2]);
            uint32_t rest = index % (updatesShape_[1] * updatesShape_[2]);
            coord1 = rest / updatesShape_[2];
            coord2 = rest % updatesShape_[2];
        } else {
            coord0 = index / (updatesShape_[1] * updatesShape_[2] * updatesShape_[3]);
            uint32_t rest = index % (updatesShape_[1] * updatesShape_[2] * updatesShape_[3]);
            coord1 = rest / (updatesShape_[2] * updatesShape_[3]);
            rest = rest % (updatesShape_[2] * updatesShape_[3]);
            coord2 = rest / updatesShape_[3];
            coord3 = rest % updatesShape_[3];
        }
    }

    __aicore__ inline bool AxisMatches(uint32_t inputDim, uint32_t weightDim, uint32_t inputCoord, uint32_t outputCoord)
    {
        if (inputDim == weightDim) return inputCoord == outputCoord;
        return outputCoord == 0;
    }

    __aicore__ inline bool InputMatchesOutput(uint32_t inCoord1, uint32_t inCoord2, uint32_t inCoord3,
        uint32_t outCoord0, uint32_t outCoord1, uint32_t outCoord2)
    {
        if (!AxisMatches(updatesShape_[1], weightsShape_[0], inCoord1, outCoord0)) return false;
        if (weightsRank_ >= 2 && !AxisMatches(updatesShape_[2], weightsShape_[1], inCoord2, outCoord1)) return false;
        if (weightsRank_ >= 3 && !AxisMatches(updatesShape_[3], weightsShape_[2], inCoord3, outCoord2)) return false;
        return true;
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceDstBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf_;
    // REDUCE_MODE_ALL 多核 map-reduce 用：SyncAll 跨核同步的 UB 工作区
    AscendC::TBuf<AscendC::TPosition::VECCALC> syncUbBuf_;
    AscendC::GlobalTensor<half> updatesHalfGm_;
    AscendC::GlobalTensor<half> daHalfGm_;
    AscendC::GlobalTensor<float> updatesFloatGm_;
    AscendC::GlobalTensor<float> daFloatGm_;
    AscendC::GlobalTensor<float> partialSumsGm_;
    AscendC::GlobalTensor<bfloat16_t> updatesBf16Gm_;
    AscendC::GlobalTensor<bfloat16_t> daBf16Gm_;
    // REDUCE_MODE_ALL 多核 map-reduce 用：workspace 中的同步区 / 各核局部和区
    AscendC::GlobalTensor<int32_t> syncGm_;
    AscendC::GlobalTensor<float> partialSumGm_;
    
    uint32_t reduceMode_ = 0;
    uint32_t dtypeMode_ = 0;
    uint32_t totalLength_ = 0;
    uint32_t outputLength_ = 0;
    uint32_t daPaddedLength_ = DA_ALIGN;
    uint32_t blockDim_ = 1;
    uint32_t coreNum_ = 1;
    uint32_t reduceSplit_ = 1;
    uint32_t sysWorkspaceSize_ = 0;
    uint32_t updatesRank_ = 0;
    uint32_t weightsRank_ = 0;
    uint32_t n_ = 1;
    uint32_t c_ = 1;
    uint32_t h_ = 1;
    uint32_t w_ = 1;
    uint32_t c1_ = 1;
    uint32_t c0_ = 1;
    uint32_t updatesShape_[MAX_RANK] = {1, 1, 1, 1};
    uint32_t weightsShape_[MAX_RANK - 1] = {1, 1, 1};
};

} // namespace NsPReluGradReduce

#endif // P_RELU_GRAD_REDUCE_H_
