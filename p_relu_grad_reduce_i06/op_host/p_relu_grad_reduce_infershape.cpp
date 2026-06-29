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
 * \file p_relu_grad_reduce_infershape.cpp
 * \brief Shape and dtype inference for PReluGradReduce.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_WEIGHTS = 2;
static constexpr int64_t IDX_UPDATES = 3;
static constexpr int64_t IDX_DA = 0;

static ge::graphStatus InferShapePReluGradReduce(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShapePReluGradReduce");

    const gert::Shape* weightsShape = context->GetInputShape(IDX_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsShape);
    gert::Shape* daShape = context->GetOutputShape(IDX_DA);
    OP_CHECK_NULL_WITH_CONTEXT(context, daShape);

    daShape->SetDimNum(weightsShape->GetDimNum());
    for (size_t i = 0; i < weightsShape->GetDimNum(); ++i) {
        daShape->SetDim(i, weightsShape->GetDim(i));
    }

    OP_LOGD(context->GetNodeName(), "End InferShapePReluGradReduce");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePReluGradReduce(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferDataTypePReluGradReduce");
    context->SetOutputDataType(IDX_DA, context->GetInputDataType(IDX_UPDATES));
    OP_LOGD(context->GetNodeName(), "End InferDataTypePReluGradReduce");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePReluGradReducePartial(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShapePReluGradReducePartial");

    const gert::Shape* updatesShape = context->GetInputShape(IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context, updatesShape);
    gert::Shape* partialShape = context->GetOutputShape(IDX_DA);
    OP_CHECK_NULL_WITH_CONTEXT(context, partialShape);

    partialShape->SetDimNum(1);
    partialShape->SetDim(0, updatesShape->GetDimNum() > 0 ? updatesShape->GetDim(0) : 1);

    OP_LOGD(context->GetNodeName(), "End InferShapePReluGradReducePartial");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePReluGradReducePartial(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferDataTypePReluGradReducePartial");
    context->SetOutputDataType(IDX_DA, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End InferDataTypePReluGradReducePartial");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePReluGradReduceFinal(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShapePReluGradReduceFinal");

    const gert::Shape* weightsShape = context->GetInputShape(IDX_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsShape);
    gert::Shape* daShape = context->GetOutputShape(IDX_DA);
    OP_CHECK_NULL_WITH_CONTEXT(context, daShape);

    daShape->SetDimNum(weightsShape->GetDimNum());
    for (size_t i = 0; i < weightsShape->GetDimNum(); ++i) {
        daShape->SetDim(i, weightsShape->GetDim(i));
    }

    OP_LOGD(context->GetNodeName(), "End InferShapePReluGradReduceFinal");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePReluGradReduceFinal(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferDataTypePReluGradReduceFinal");
    context->SetOutputDataType(IDX_DA, context->GetInputDataType(IDX_WEIGHTS));
    OP_LOGD(context->GetNodeName(), "End InferDataTypePReluGradReduceFinal");
    return GRAPH_SUCCESS;
}

} // 【关键修改1】：必须在这里先闭合 namespace ops！

// 【关键修改2】：在全局作用域调用标准的 IMPL_OP 宏，并给函数加上 ops:: 前缀
IMPL_OP(PReluGradReduce)
    .InferShape(ops::InferShapePReluGradReduce)
    .InferDataType(ops::InferDataTypePReluGradReduce);

IMPL_OP(PReluGradReducePartial)
    .InferShape(ops::InferShapePReluGradReducePartial)
    .InferDataType(ops::InferDataTypePReluGradReducePartial);

IMPL_OP(PReluGradReduceFinal)
    .InferShape(ops::InferShapePReluGradReduceFinal)
    .InferDataType(ops::InferDataTypePReluGradReduceFinal);
