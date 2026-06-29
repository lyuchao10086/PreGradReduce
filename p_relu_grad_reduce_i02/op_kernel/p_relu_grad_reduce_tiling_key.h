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
 * \file p_relu_grad_reduce_tiling_key.h
 * \brief Tiling key declaration for PReluGradReduce.
 */
#ifndef P_RELU_GRAD_REDUCE_TILING_KEY_H_
#define P_RELU_GRAD_REDUCE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// 补齐三种数据类型的 Key
#define P_RELU_GRAD_REDUCE_TILING_KEY_FP16 0
#define P_RELU_GRAD_REDUCE_TILING_KEY_FP32 1
#define P_RELU_GRAD_REDUCE_TILING_KEY_BF16 2

ASCENDC_TPL_ARGS_DECL(PReluGradReduce,
    ASCENDC_TPL_UINT_DECL(schMode, 3, ASCENDC_TPL_UI_LIST, 
        P_RELU_GRAD_REDUCE_TILING_KEY_FP16, 
        P_RELU_GRAD_REDUCE_TILING_KEY_FP32, 
        P_RELU_GRAD_REDUCE_TILING_KEY_BF16)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 
            P_RELU_GRAD_REDUCE_TILING_KEY_FP16, 
            P_RELU_GRAD_REDUCE_TILING_KEY_FP32, 
            P_RELU_GRAD_REDUCE_TILING_KEY_BF16)));

#endif // P_RELU_GRAD_REDUCE_TILING_KEY_H_
