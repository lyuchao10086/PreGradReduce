/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file p_relu_grad_reduce_tiling_data.h
 * \brief Tiling data for PReluGradReduce.
 */
#ifndef P_RELU_GRAD_REDUCE_TILING_DATA_H_
#define P_RELU_GRAD_REDUCE_TILING_DATA_H_

#include <stdint.h>

struct PReluGradReduceTilingData {
    uint32_t reduceMode = 0;
    uint32_t dtypeMode = 0;
    uint32_t totalLength = 0;
    uint32_t outputLength = 0;
    uint32_t daPaddedLength = 0;
    uint32_t blockDim = 1;
    uint32_t updatesRank = 0;
    uint32_t weightsRank = 0;
    uint32_t n = 1;
    uint32_t c = 1;
    uint32_t h = 1;
    uint32_t w = 1;
    uint32_t c1 = 1;
    uint32_t c0 = 1;
    uint32_t updatesShape0 = 1;
    uint32_t updatesShape1 = 1;
    uint32_t updatesShape2 = 1;
    uint32_t updatesShape3 = 1;
    uint32_t weightsShape0 = 1;
    uint32_t weightsShape1 = 1;
    uint32_t weightsShape2 = 1;
    uint32_t sysWorkspaceSize = 0;
    // 运行时实际参与计算的核数（REDUCE_MODE_ALL 多核 map-reduce 用，
    // kernel 侧据此切分 totalLength 并确定 workspace 中 sync/partialSum 区的大小）
    uint32_t coreNum = 1;
    // NCHW/NC1HWC0 reduce-split 使用：每个输出 channel/C1 在 reduce 维上拆成多少个 partial slice。
    uint32_t reduceSplit = 1;

    // uint32_t padding1 = 0;
};

#endif // P_RELU_GRAD_REDUCE_TILING_DATA_H_
