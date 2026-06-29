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
 * \file aclnn_p_relu_grad_reduce.cpp
 * \brief PReluGradReduce 的 aclnn API 层
 *
 * 算子语义：
 *   输入 updates  —— 已经过掩码处理的上游梯度（shape 可以是 ND 或 NC1HWC0）
 *   输入 weights  —— PReLU 权重 alpha（决定 reduction 的 channel 对齐方式）
 *   输出 da       —— alpha 的梯度，形状与 weights 相同
 *
 * 本文件对应 libcust_opapi.so 中需要导出的两个符号：
 *   aclnnPReluGradReduceGetWorkspaceSize
 *   aclnnPReluGradReduce
 */

#include "aclnn_p_relu_grad_reduce.h"
// ↓ 声明 l0op::PReluGradReduce 的头文件。
//   如果项目里还没有这个文件，需要新建一个，见文件末尾的说明。
#include "p_relu_grad_reduce_l0op.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────────────────────
//  全局常量
// ─────────────────────────────────────────────────────────────────────────
namespace {
// 与 tiling 中 DTYPE_MODE_FP16 / FP32 / BF16 保持一致
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16,
    op::DataType::DT_FLOAT,
    op::DataType::DT_BF16
};
} // namespace

// ─────────────────────────────────────────────────────────────────────────
//  1. 空指针检查
// ─────────────────────────────────────────────────────────────────────────
static bool CheckNotNull(const aclTensor *updates,
                         const aclTensor *weights,
                         const aclTensor *da)
{
    OP_CHECK_NULL(updates, return false);
    OP_CHECK_NULL(weights, return false);
    OP_CHECK_NULL(da,      return false);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
//  2. Format 检查
//     允许两种组合：
//       (a) 全部 ND
//       (b) updates/da 为 NC1HWC0，weights 为 ND 或 NC1HWC0
// ─────────────────────────────────────────────────────────────────────────
static bool CheckFormatValid(const aclTensor *updates,
                             const aclTensor *weights,
                             const aclTensor *da)
{
    const auto updatesFormat = updates->GetStorageFormat();
    const auto weightsFormat = weights->GetStorageFormat();
    const auto daFormat      = da->GetStorageFormat();

    // updates 只允许 ND 或 NC1HWC0
    if (updatesFormat != Format::FORMAT_ND && updatesFormat != Format::FORMAT_NC1HWC0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "updates format(%d) only supports ND or NC1HWC0",
            static_cast<int>(updatesFormat));
        return false;
    }

    // da 必须与 updates format 一致
    if (daFormat != updatesFormat) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "da format(%d) must match updates format(%d)",
            static_cast<int>(daFormat), static_cast<int>(updatesFormat));
        return false;
    }

    // NC1HWC0 分支：weights 只允许 ND 或 NC1HWC0（tiling 已对两种形状做兼容）
    if (updatesFormat == Format::FORMAT_NC1HWC0) {
        if (weightsFormat != Format::FORMAT_ND && weightsFormat != Format::FORMAT_NC1HWC0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "NC1HWC0 branch: weights format(%d) must be ND or NC1HWC0",
                static_cast<int>(weightsFormat));
            return false;
        }
    } else {
        // ND 分支：weights 也必须是 ND
        if (weightsFormat != Format::FORMAT_ND) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ND branch: weights format(%d) must be ND",
                static_cast<int>(weightsFormat));
            return false;
        }
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────
//  3. DataType 检查
//     所有张量 dtype 必须一致，且在支持列表内
// ─────────────────────────────────────────────────────────────────────────
static bool CheckDtypeValid(const aclTensor *updates,
                            const aclTensor *weights,
                            const aclTensor *da)
{
    // updates 作为基准 dtype
    OP_CHECK_DTYPE_NOT_SUPPORT(updates, DTYPE_SUPPORT_LIST, return false);
    const auto baseDtype = updates->GetDataType();

    if (weights->GetDataType() != baseDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "weights dtype(%s) inconsistent with updates dtype(%s)",
            op::ToString(weights->GetDataType()).GetString(),
            op::ToString(baseDtype).GetString());
        return false;
    }

    if (da->GetDataType() != baseDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "da dtype(%s) inconsistent with updates dtype(%s)",
            op::ToString(da->GetDataType()).GetString(),
            op::ToString(baseDtype).GetString());
        return false;
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────
//  4. Shape 检查
//     da 的 shape 必须与 weights 完全相同（da 是 weights 的梯度）
// ─────────────────────────────────────────────────────────────────────────
static bool CheckShapeValid(const aclTensor *updates,
                            const aclTensor *weights,
                            const aclTensor *da)
{
    const auto updatesShape = updates->GetViewShape();
    const auto weightsShape = weights->GetViewShape();
    const auto daShape      = da->GetViewShape();

    if (updatesShape.GetDimNum() < 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "updates must have at least 1 dimension");
        return false;
    }

    // da 和 weights 必须同 rank
    if (daShape.GetDimNum() != weightsShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "da rank(%zu) must equal weights rank(%zu)",
            daShape.GetDimNum(), weightsShape.GetDimNum());
        return false;
    }

    // da 和 weights 逐维度相同
    for (size_t i = 0; i < weightsShape.GetDimNum(); ++i) {
        if (daShape.GetDim(i) != weightsShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "da/weights shape mismatch at dim%zu: da=%ld weights=%ld",
                i, daShape.GetDim(i), weightsShape.GetDim(i));
            return false;
        }
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────
//  汇总检查
// ─────────────────────────────────────────────────────────────────────────
static aclnnStatus CheckParams(const aclTensor *updates,
                               const aclTensor *weights,
                               const aclTensor *da)
{
    // 顺序：nullptr → dtype → shape → format
    CHECK_RET(CheckNotNull(updates, weights, da),     ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(updates, weights, da),  ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(updates, weights, da),  ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(updates, weights, da), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────
//  对外 API 函数 1：计算 workspace 大小，同时构建执行计划
// ─────────────────────────────────────────────────────────────────────────
aclnnStatus aclnnPReluGradReduceGetWorkspaceSize(
    const aclTensor *grads,
    const aclTensor *features,
    const aclTensor *weights,
    const aclTensor *updates,
    aclTensor       *da,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
{
    (void)grads;
    (void)features;

    L2_DFX_PHASE_1(aclnnPReluGradReduce,
        DFX_IN(updates, weights),
        DFX_OUT(da));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(updates, weights, da);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (updates->IsEmpty() || weights->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto updatesContiguous = l0op::Contiguous(updates, uniqueExecutor.get());
    CHECK_RET(updatesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto weightsContiguous = l0op::Contiguous(weights, uniqueExecutor.get());
    CHECK_RET(weightsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto daTemp = l0op::PReluGradReduce(updatesContiguous, weightsContiguous,
                                         uniqueExecutor.get());
    CHECK_RET(daTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto copyResult = l0op::ViewCopy(daTemp, da, uniqueExecutor.get());
    CHECK_RET(copyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────
//  对外 API 函数 2：提交执行（固定模板写法，无需改动）
// ─────────────────────────────────────────────────────────────────────────
aclnnStatus aclnnPReluGradReduce(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
{
    L2_DFX_PHASE_2(aclnnPReluGradReduce);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

/*
 * ═══════════════════════════════════════════════════════════════════════════
 *  附：需要同时新建 p_relu_grad_reduce_l0op.h
 *  （放在 op_api/ 目录，或 op_api/include/ 里）
 *
 *  内容模板如下，对照 thnn_fused_lstm_cell_grad.h 的格式写：
 *
 *  #pragma once
 *  #include "opdev/op_executor.h"
 *  #include "aclnn/acl_tensor.h"
 *
 *  namespace l0op {
 *  // 输入：updates (already-masked gradient), weights (alpha)
 *  // 输出：临时 da tensor，由框架管理生命周期
 *  aclTensor* PReluGradReduce(const aclTensor* updates,
 *                              const aclTensor* weights,
 *                              opdev::OpExecutor* executor);
 *  } // namespace l0op
 *
 *  同时需要在 op_api/CMakeLists.txt 里把
 *  aclnn_p_relu_grad_reduce.cpp 加到编译列表，才能链接进 libcust_opapi.so。
 * ═══════════════════════════════════════════════════════════════════════════
 */