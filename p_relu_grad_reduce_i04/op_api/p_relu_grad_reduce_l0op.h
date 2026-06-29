#pragma once
#include "opdev/op_executor.h"
#include "aclnn/acl_tensor.h"

namespace l0op {
// 输入：updates, weights (alpha)
// 输出：临时 da tensor
aclTensor* PReluGradReduce(const aclTensor* updates,
                           const aclTensor* weights,
                           opdev::OpExecutor* executor);
} // namespace l0op