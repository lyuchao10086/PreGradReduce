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
 * \file p_relu_grad_reduce_def.cpp
 * \brief PReluGradReduce op definition for the ops-nn custom package.
 */
#include <initializer_list>
#include "register/op_def_registry.h"

namespace ops {

const std::initializer_list<ge::DataType> GRAD_DTYPES = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16
};
const std::initializer_list<ge::Format> GRAD_FORMATS = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0
};
const std::initializer_list<ge::Format> WEIGHT_FORMATS = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0
};

class PReluGradReduce : public OpDef {
public:
    explicit PReluGradReduce(const char* name) : OpDef(name)
    {
        this->Input("grads")
            .ParamType(REQUIRED)
            .DataType(GRAD_DTYPES)
            .Format(GRAD_FORMATS)
            .UnknownShapeFormat(GRAD_FORMATS)
            .AutoContiguous();
        this->Input("features")
            .ParamType(REQUIRED)
            .DataType(GRAD_DTYPES)
            .Format(GRAD_FORMATS)
            .UnknownShapeFormat(GRAD_FORMATS)
            .AutoContiguous();
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType(GRAD_DTYPES)
            .Format(WEIGHT_FORMATS)
            .UnknownShapeFormat(WEIGHT_FORMATS)
            .AutoContiguous();
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType(GRAD_DTYPES)
            .Format(GRAD_FORMATS)
            .UnknownShapeFormat(GRAD_FORMATS)
            .AutoContiguous();
        this->Output("da")
            .ParamType(REQUIRED)
            .DataType(GRAD_DTYPES)
            .Format(WEIGHT_FORMATS)
            .UnknownShapeFormat(WEIGHT_FORMATS)
            .AutoContiguous();


        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "p_relu_grad_reduce")
            .ExtendCfgInfo("opInterface.value", "p_relu_grad_reduce");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(PReluGradReduce);
} // namespace ops
