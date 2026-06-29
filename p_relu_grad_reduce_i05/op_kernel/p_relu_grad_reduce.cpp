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
 * \file p_relu_grad_reduce.cpp
 * \brief PReluGradReduce AI Core kernel entry.
 */

#include "p_relu_grad_reduce.h"
#include "p_relu_grad_reduce_tiling_key.h"

template <uint32_t schMode>
__global__ __aicore__ void p_relu_grad_reduce(
    GM_ADDR grads,
    GM_ADDR features,
    GM_ADDR weights,
    GM_ADDR updates,
    GM_ADDR da,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)grads;
    (void)features;
    (void)weights;

    AscendC::SetSysWorkspace(workspace);


    REGISTER_TILING_DEFAULT(PReluGradReduceTilingData);
    GET_TILING_DATA_WITH_STRUCT(PReluGradReduceTilingData, tilingData, tiling);

    // 开放 FP16, FP32, BF16 的宏分支
    if (schMode == P_RELU_GRAD_REDUCE_TILING_KEY_FP16 ||
        schMode == P_RELU_GRAD_REDUCE_TILING_KEY_FP32 ||
        schMode == P_RELU_GRAD_REDUCE_TILING_KEY_BF16) {
        NsPReluGradReduce::PReluGradReduce op;
        op.Init(updates, da, workspace, &tilingData);
        op.Process();
    }
}
