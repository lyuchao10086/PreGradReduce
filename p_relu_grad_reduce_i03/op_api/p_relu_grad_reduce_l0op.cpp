#include "p_relu_grad_reduce_l0op.h"
#include "opdev/make_op_executor.h"

namespace l0op {

aclTensor* PReluGradReduce(const aclTensor* updates,
                           const aclTensor* weights,
                           opdev::OpExecutor* executor)
{
    // 调用底层算子名称，这个名称必须和 def 文件里的注册名一致
    // 构造算子执行器
    auto callOp = opdev::MakeOpExecutor("PReluGradReduce");

    // 绑定输入
    callOp->AddInput("grads", updates);    // 内部实现可以先用 updates 顶替
    callOp->AddInput("features", updates); // 内部实现可以先用 updates 顶替
    callOp->AddInput("weights", weights);
    callOp->AddInput("updates", updates);

    // 绑定输出
    // 让框架根据 InferShape 和 InferDataType 自动推导和分配 da 张量
    aclTensor* da = callOp->AddOutput("da");

    // 挂载到当前的执行上下文
    callOp->MountTo(executor);

    return da;
}

} // namespace l0op