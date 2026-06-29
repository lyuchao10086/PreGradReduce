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
 * \file p_relu_grad_reduce_proto.h
 * \brief PReluGradReduce operator prototype.
 */
#ifndef OPS_OP_PROTO_INC_P_RELU_GRAD_REDUCE_H_
#define OPS_OP_PROTO_INC_P_RELU_GRAD_REDUCE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Reduces the intermediate weight-gradient tensor in PReLU backward.
*@par Inputs:
* @li grads: Reverse gradient of PReLU forward output.
* @li features: PReLU forward input.
* @li weights: PReLU weight tensor. The output shape follows this input.
* @li updates: Intermediate weight-gradient tensor to be reduced.
*@par Outputs:
* @li da: Reduced PReLU weight gradient.
*/
REG_OP(PReluGradReduce)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(weights, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(da, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(PReluGradReduce)

REG_OP(PReluGradReducePartial)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(weights, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(updates, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(partial, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(PReluGradReducePartial)

REG_OP(PReluGradReduceFinal)
    .INPUT(grads, TensorType({DT_FLOAT}))
    .INPUT(features, TensorType({DT_FLOAT}))
    .INPUT(weights, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(updates, TensorType({DT_FLOAT}))
    .OUTPUT(da, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(PReluGradReduceFinal)

} // namespace ge

#endif // OPS_OP_PROTO_INC_P_RELU_GRAD_REDUCE_H_
