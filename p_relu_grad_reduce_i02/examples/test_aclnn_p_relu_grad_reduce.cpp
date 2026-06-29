/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn_p_relu_grad_reduce.h"

extern "C" void* dlopen(const char* filename, int flags) __attribute__((weak));
extern "C" void* dlsym(void* handle, const char* symbol) __attribute__((weak));
extern "C" char* dlerror(void) __attribute__((weak));

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)                  \
    do {                                         \
        fprintf(stderr, message, ##__VA_ARGS__); \
        fflush(stderr);                          \
    } while (0)

constexpr int RTLD_NOW = 2;
constexpr int RTLD_GLOBAL = 0x100;
constexpr size_t kSamplePrintSize = 8;
constexpr float kFp16Tolerance = 1.0e-2F;
constexpr float kFp32Tolerance = 1.0e-4F;
constexpr float kFp16RelTolerance = 1.0e-3F;
constexpr float kFp32RelTolerance = 1.0e-5F;

using AclopCompileAndExecuteFunc = aclError (*)(const char* opType, int numInputs,
    aclTensorDesc* const inputDesc[], aclDataBuffer* const inputs[], int numOutputs,
    aclTensorDesc* const outputDesc[], aclDataBuffer* const outputs[], aclopAttr* attr, aclopEngineType engineType,
    aclopCompileType compileFlag, const char* opPath, aclrtStream stream);
using AclnnPReluGradReduceGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor* grads,
    const aclTensor* features, const aclTensor* weights, const aclTensor* updates, aclTensor* da,
    uint64_t* workspaceSize, aclOpExecutor** executor);
using AclnnPReluGradReduceFunc = aclnnStatus (*)(void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);

struct AclnnPReluGradReduceFuncs {
    AclnnPReluGradReduceGetWorkspaceSizeFunc getWorkspaceSize = nullptr;
    AclnnPReluGradReduceFunc run = nullptr;
};

enum class InputPattern {
    RAMP,
    SIGNED_PATTERN,
    CANCEL_PATTERN
};

struct CaseSpec {
    const char* name;
    std::vector<int64_t> updatesShape;
    std::vector<int64_t> weightsShape;
    aclDataType dataType;
    aclFormat updatesFormat;
    aclFormat weightsFormat;
    aclFormat daFormat;
    bool allOnes;
    bool benchmark;
    float inputScale;
    const char* reportNote;
    InputPattern inputPattern = InputPattern::RAMP;
};

std::string JoinPath(const char* base, const std::string& suffix)
{
    if (base == nullptr || base[0] == '\0') {
        return "";
    }
    return std::string(base) + suffix;
}

std::string JoinPath(const std::string& base, const std::string& name)
{
    if (base.empty() || name.empty()) {
        return "";
    }
    return base.back() == '/' ? base + name : base + "/" + name;
}

bool IsDirectory(const std::string& path)
{
    struct stat info;
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

bool IsTargetOpApiLibrary(const std::string& name)
{
    bool isSharedObject = name.size() > 3 && name.substr(name.size() - 3) == ".so";
    return name == "libcust_opapi.so" || name == "libcustom_opapi.so" || name == "libopapi.so" ||
           (isSharedObject && name.find("opapi") != std::string::npos);
}

void AddCandidate(std::vector<std::string>& candidates, const std::string& path)
{
    if (!path.empty() && std::find(candidates.begin(), candidates.end(), path) == candidates.end()) {
        candidates.emplace_back(path);
    }
}

std::string GetCurrentDir()
{
    char buffer[4096] = {0};
    return getcwd(buffer, sizeof(buffer)) == nullptr ? "" : std::string(buffer);
}

std::string ParentDir(const std::string& path)
{
    if (path.empty() || path == "/") {
        return "";
    }
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return "";
    }
    return pos == 0 ? "/" : path.substr(0, pos);
}

void CollectOpApiLibraries(const std::string& dir, int depth, std::vector<std::string>& candidates)
{
    if (dir.empty() || depth < 0 || !IsDirectory(dir)) {
        return;
    }

    DIR* handle = opendir(dir.c_str());
    if (handle == nullptr) {
        return;
    }

    std::vector<std::string> childDirs;
    struct dirent* entry = nullptr;
    while ((entry = readdir(handle)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        std::string path = JoinPath(dir, name);
        if (IsTargetOpApiLibrary(name)) {
            AddCandidate(candidates, path);
        } else if (depth > 0 && IsDirectory(path)) {
            childDirs.emplace_back(path);
        }
    }
    closedir(handle);

    for (const auto& child : childDirs) {
        CollectOpApiLibraries(child, depth - 1, candidates);
    }
}

uint32_t GetEnvUint32(const char* name, uint32_t defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
}

bool IsBenchmarkEnabled()
{
    const char* value = std::getenv("PRELU_GRAD_REDUCE_BENCH");
    return value != nullptr && std::string(value) == "1";
}
const char* DataTypeName(aclDataType dataType)
{
    if (dataType == ACL_FLOAT) {
        return "float32";
    }
    if (dataType == ACL_FLOAT16) {
        return "float16";
    }
    if (dataType == ACL_BF16) {
        return "bfloat16";
    }
    return "unknown";
}

std::string ShapeToString(const std::vector<int64_t>& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i + 1 != shape.size()) {
            oss << ",";
        }
    }
    oss << "]";
    return oss.str();
}

AclopCompileAndExecuteFunc LoadAclopCompileAndExecute()
{
    if (dlopen == nullptr || dlsym == nullptr) {
        LOG_PRINT("dlopen/dlsym is unavailable. Cannot load aclopCompileAndExecute dynamically.\n");
        return nullptr;
    }

    std::vector<std::string> candidates;
    candidates.emplace_back("libacl_op_compiler.so");
    candidates.emplace_back("libascendcl.so");
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_HOME_PATH"), "/lib64/libacl_op_compiler.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_HOME_PATH"), "/runtime/lib64/libacl_op_compiler.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_HOME_PATH"), "/compiler/lib64/libacl_op_compiler.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_HOME_PATH"), "/lib64/libascendcl.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_HOME_PATH"), "/runtime/lib64/libascendcl.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_TOOLKIT_HOME"), "/lib64/libacl_op_compiler.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_TOOLKIT_HOME"), "/runtime/lib64/libacl_op_compiler.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_TOOLKIT_HOME"), "/compiler/lib64/libacl_op_compiler.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_TOOLKIT_HOME"), "/lib64/libascendcl.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_TOOLKIT_HOME"), "/runtime/lib64/libascendcl.so"));

    for (const auto& lib : candidates) {
        if (lib.empty()) {
            continue;
        }
        void* handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle == nullptr) {
            continue;
        }
        auto func = reinterpret_cast<AclopCompileAndExecuteFunc>(dlsym(handle, "aclopCompileAndExecute"));
        if (func != nullptr) {
            LOG_PRINT("Loaded aclopCompileAndExecute from %s\n", lib.c_str());
            return func;
        }
    }

    const char* err = dlerror == nullptr ? nullptr : dlerror();
    LOG_PRINT("Failed to load aclopCompileAndExecute.%s%s\n",
        err == nullptr ? "" : " Last dlerror: ", err == nullptr ? "" : err);
    return nullptr;
}

AclnnPReluGradReduceFuncs LoadAclnnPReluGradReduce()
{
    AclnnPReluGradReduceFuncs funcs;
    if (dlopen == nullptr || dlsym == nullptr) {
        LOG_PRINT("dlopen/dlsym is unavailable. Cannot load aclnn PReluGradReduce dynamically.\n");
        return funcs;
    }

    std::vector<std::string> candidates;
    candidates.emplace_back("");
    AddCandidate(candidates, std::getenv("PRELU_GRAD_REDUCE_OPAPI_LIB") == nullptr ?
        "" : std::getenv("PRELU_GRAD_REDUCE_OPAPI_LIB"));
    candidates.emplace_back("libcust_opapi.so");
    candidates.emplace_back("libcustom_opapi.so");
    candidates.emplace_back("libopapi.so");
    std::string cwd = GetCurrentDir();
    std::string parent = ParentDir(cwd);
    CollectOpApiLibraries(cwd, 5, candidates);
    CollectOpApiLibraries(JoinPath(cwd, "build"), 5, candidates);
    CollectOpApiLibraries(parent, 3, candidates);
    CollectOpApiLibraries(JoinPath(parent, "build"), 5, candidates);
    CollectOpApiLibraries("/mnt/workspace/gitCode/cann/ops-nn/build", 5, candidates);
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_HOME_PATH"), "/lib64/libcust_opapi.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_HOME_PATH"), "/opp/vendors/custom/op_api/lib/libcust_opapi.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_HOME_PATH"), "/opp/vendors/custom_nn/op_api/lib/libcust_opapi.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_TOOLKIT_HOME"), "/lib64/libcust_opapi.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_TOOLKIT_HOME"), "/opp/vendors/custom/op_api/lib/libcust_opapi.so"));
    candidates.emplace_back(JoinPath(std::getenv("ASCEND_TOOLKIT_HOME"), "/opp/vendors/custom_nn/op_api/lib/libcust_opapi.so"));
    candidates.emplace_back("/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom/op_api/lib/libcust_opapi.so");
    candidates.emplace_back("/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_nn/op_api/lib/libcust_opapi.so");

    for (const auto& lib : candidates) {
        void* handle = lib.empty() ? dlopen(nullptr, RTLD_NOW | RTLD_GLOBAL) : dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle == nullptr) {
            continue;
        }

        // === 【修改的核心逻辑：双重查找符号】 ===
        // 1. 先尝试找带 Custom 后缀的函数名
        funcs.getWorkspaceSize = reinterpret_cast<AclnnPReluGradReduceGetWorkspaceSizeFunc>(
            dlsym(handle, "aclnnPReluGradReduceCustomGetWorkspaceSize"));
        funcs.run = reinterpret_cast<AclnnPReluGradReduceFunc>(
            dlsym(handle, "aclnnPReluGradReduceCustom"));

        // 2. 如果没找到，退一步尝试找不带 Custom 后缀的函数名
        if (funcs.getWorkspaceSize == nullptr) {
            funcs.getWorkspaceSize = reinterpret_cast<AclnnPReluGradReduceGetWorkspaceSizeFunc>(
                dlsym(handle, "aclnnPReluGradReduceGetWorkspaceSize"));
        }
        if (funcs.run == nullptr) {
            funcs.run = reinterpret_cast<AclnnPReluGradReduceFunc>(
                dlsym(handle, "aclnnPReluGradReduce"));
        }
        // === 【核心逻辑结束】 ===

        if (funcs.getWorkspaceSize != nullptr && funcs.run != nullptr) {
            LOG_PRINT("Loaded aclnn PReluGradReduce from %s\n", lib.empty() ? "<process>" : lib.c_str());
            return funcs;
        }
        
        if (!lib.empty()) {
            LOG_PRINT("Opened %s, but missing symbols: %s%s%s\n", lib.c_str(),
                funcs.getWorkspaceSize == nullptr ? "aclnnPReluGradReduce(Custom)GetWorkspaceSize " : "",
                funcs.run == nullptr ? "aclnnPReluGradReduce(Custom)" : "",
                funcs.getWorkspaceSize != nullptr && funcs.run != nullptr ? "none" : "");
        }
    }

    const char* err = dlerror == nullptr ? nullptr : dlerror();
    LOG_PRINT("Failed to load aclnn PReluGradReduce symbols.%s%s\n",
        err == nullptr ? "" : " Last dlerror: ", err == nullptr ? "" : err);
    LOG_PRINT("Tried aclnn opapi candidates:\n");
    for (const auto& lib : candidates) {
        LOG_PRINT("  %s\n", lib.empty() ? "<process>" : lib.c_str());
    }
    funcs.getWorkspaceSize = nullptr;
    funcs.run = nullptr;
    return funcs;
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
    uint32_t mant = bits & 0x7fffffU;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000U) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000U) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000U) >> 13));
}

float HalfToFloat(uint16_t value)
{
    uint32_t sign = (static_cast<uint32_t>(value & 0x8000U)) << 16;
    uint32_t exp = (value >> 10) & 0x1fU;
    uint32_t mant = value & 0x03ffU;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffU;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000U | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t roundingBias = ((bits >> 16) & 1U) + 0x7fffU;
    return static_cast<uint16_t>((bits + roundingBias) >> 16);
}

float BFloat16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    LOG_PRINT("[INIT] aclInit\n");
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("[INIT] aclrtSetDevice device=%d\n", deviceId);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("[INIT] aclrtCreateStream\n");
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("[INIT] done\n");
    return ACL_SUCCESS;
}

aclError DestroyDataBuffer(aclDataBuffer* dataBuffer)
{
    if (dataBuffer == nullptr) {
        return ACL_SUCCESS;
    }
    return aclDestroyDataBuffer(dataBuffer);
}

void DestroyTensorDesc(aclTensorDesc* tensorDesc)
{
    if (tensorDesc != nullptr) {
        aclDestroyTensorDesc(tensorDesc);
    }
}

void DestroyAclTensor(aclTensor* tensor)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
}

std::vector<int64_t> MakeContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

template <typename T>
int CreateDeviceBuffer(const std::vector<T>& hostData, void** deviceAddr, aclDataBuffer** dataBuffer)
{
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    *dataBuffer = aclCreateDataBuffer(*deviceAddr, size);
    CHECK_RET(*dataBuffer != nullptr, LOG_PRINT("aclCreateDataBuffer failed.\n"); return 1);
    return ACL_SUCCESS;
}

aclTensorDesc* CreateTensorDesc(const std::vector<int64_t>& shape, aclDataType dataType, aclFormat format)
{
    aclTensorDesc* desc = aclCreateTensorDesc(dataType, static_cast<int>(shape.size()), shape.data(), format);
    if (desc == nullptr) {
        return nullptr;
    }
    aclError ret = aclSetTensorOriginShape(desc, static_cast<int>(shape.size()), shape.data());
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclSetTensorOriginShape failed. ERROR: %d\n", ret);
        aclDestroyTensorDesc(desc);
        return nullptr;
    }
    ret = aclSetTensorOriginFormat(desc, ACL_FORMAT_ND);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclSetTensorOriginFormat failed. ERROR: %d\n", ret);
        aclDestroyTensorDesc(desc);
        return nullptr;
    }
    return desc;
}

aclTensor* CreateAclTensor(const std::vector<int64_t>& shape, aclDataType dataType, aclFormat format, void* deviceAddr)
{
    std::vector<int64_t> strides = MakeContiguousStrides(shape);
    return aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
        shape.data(), shape.size(), deviceAddr);
}

aclError RunPReluGradReduceByAclnn(aclDataBuffer* gradsBuffer, aclDataBuffer* featuresBuffer,
    aclDataBuffer* weightsBuffer, aclDataBuffer* updatesBuffer, aclDataBuffer* daBuffer, const CaseSpec& testCase,
    aclrtStream stream)
{
    static AclnnPReluGradReduceFuncs aclnnFuncs = LoadAclnnPReluGradReduce();
    CHECK_RET(aclnnFuncs.getWorkspaceSize != nullptr && aclnnFuncs.run != nullptr,
        return static_cast<aclError>(1));

    void* gradsAddr = aclGetDataBufferAddr(gradsBuffer);
    void* featuresAddr = aclGetDataBufferAddr(featuresBuffer);
    void* weightsAddr = aclGetDataBufferAddr(weightsBuffer);
    void* updatesAddr = aclGetDataBufferAddr(updatesBuffer);
    void* daAddr = aclGetDataBufferAddr(daBuffer);
    CHECK_RET(gradsAddr != nullptr && featuresAddr != nullptr && weightsAddr != nullptr &&
                  updatesAddr != nullptr && daAddr != nullptr,
        LOG_PRINT("aclGetDataBufferAddr failed.\n");
        return static_cast<aclError>(1));

    aclTensor* gradsTensor = CreateAclTensor(testCase.updatesShape, testCase.dataType, testCase.updatesFormat, gradsAddr);
    aclTensor* featuresTensor =
        CreateAclTensor(testCase.updatesShape, testCase.dataType, testCase.updatesFormat, featuresAddr);
    aclTensor* weightsTensor =
        CreateAclTensor(testCase.weightsShape, testCase.dataType, testCase.weightsFormat, weightsAddr);
    aclTensor* updatesTensor =
        CreateAclTensor(testCase.updatesShape, testCase.dataType, testCase.updatesFormat, updatesAddr);
    aclTensor* daTensor = CreateAclTensor(testCase.weightsShape, testCase.dataType, testCase.daFormat, daAddr);
    std::unique_ptr<aclTensor, void (*)(aclTensor*)> gradsTensorPtr(gradsTensor, DestroyAclTensor);
    std::unique_ptr<aclTensor, void (*)(aclTensor*)> featuresTensorPtr(featuresTensor, DestroyAclTensor);
    std::unique_ptr<aclTensor, void (*)(aclTensor*)> weightsTensorPtr(weightsTensor, DestroyAclTensor);
    std::unique_ptr<aclTensor, void (*)(aclTensor*)> updatesTensorPtr(updatesTensor, DestroyAclTensor);
    std::unique_ptr<aclTensor, void (*)(aclTensor*)> daTensorPtr(daTensor, DestroyAclTensor);
    CHECK_RET(gradsTensor != nullptr && featuresTensor != nullptr && weightsTensor != nullptr &&
                  updatesTensor != nullptr && daTensor != nullptr,
        LOG_PRINT("aclCreateTensor failed.\n");
        return static_cast<aclError>(1));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto nnRet = aclnnFuncs.getWorkspaceSize(gradsTensor, featuresTensor, weightsTensor, updatesTensor,
        daTensor, &workspaceSize, &executor);
    CHECK_RET(nnRet == ACL_SUCCESS,
        LOG_PRINT("aclnnPReluGradReduceGetWorkspaceSize failed, case=%s ERROR: %d\n", testCase.name, nnRet);
        return static_cast<aclError>(nnRet));

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    
        ret = aclrtMemset(workspace, workspaceSize, 0, workspaceSize);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemset workspace failed. ERROR: %d\n", ret); return ret);
    }
    std::unique_ptr<void, aclError (*)(void*)> workspacePtr(workspace, aclrtFree);

    nnRet = aclnnFuncs.run(workspace, workspaceSize, executor, stream);
    CHECK_RET(nnRet == ACL_SUCCESS,
        LOG_PRINT("aclnnPReluGradReduce failed, case=%s ERROR: %d\n", testCase.name, nnRet);
        return static_cast<aclError>(nnRet));
    return ACL_SUCCESS;
}

aclError RunPReluGradReduceByAclop(aclDataBuffer* gradsBuffer, aclDataBuffer* featuresBuffer,
    aclDataBuffer* weightsBuffer, aclDataBuffer* updatesBuffer, aclDataBuffer* daBuffer, const CaseSpec& testCase,
    aclrtStream stream)
{
    static AclopCompileAndExecuteFunc aclopCompileAndExecuteFunc = LoadAclopCompileAndExecute();
    CHECK_RET(aclopCompileAndExecuteFunc != nullptr, return static_cast<aclError>(1));

    aclTensorDesc* gradsDesc = CreateTensorDesc(testCase.updatesShape, testCase.dataType, testCase.updatesFormat);
    aclTensorDesc* featuresDesc = CreateTensorDesc(testCase.updatesShape, testCase.dataType, testCase.updatesFormat);
    aclTensorDesc* weightsDesc = CreateTensorDesc(testCase.weightsShape, testCase.dataType, testCase.weightsFormat);
    aclTensorDesc* updatesDesc = CreateTensorDesc(testCase.updatesShape, testCase.dataType, testCase.updatesFormat);
    aclTensorDesc* daDesc = CreateTensorDesc(testCase.weightsShape, testCase.dataType, testCase.daFormat);
    std::unique_ptr<aclTensorDesc, void (*)(aclTensorDesc*)> gradsDescPtr(gradsDesc, DestroyTensorDesc);
    std::unique_ptr<aclTensorDesc, void (*)(aclTensorDesc*)> featuresDescPtr(featuresDesc, DestroyTensorDesc);
    std::unique_ptr<aclTensorDesc, void (*)(aclTensorDesc*)> weightsDescPtr(weightsDesc, DestroyTensorDesc);
    std::unique_ptr<aclTensorDesc, void (*)(aclTensorDesc*)> updatesDescPtr(updatesDesc, DestroyTensorDesc);
    std::unique_ptr<aclTensorDesc, void (*)(aclTensorDesc*)> daDescPtr(daDesc, DestroyTensorDesc);
    CHECK_RET(gradsDesc != nullptr && featuresDesc != nullptr && weightsDesc != nullptr && updatesDesc != nullptr &&
                  daDesc != nullptr,
        LOG_PRINT("aclCreateTensorDesc failed.\n");
        return static_cast<aclError>(1));

    aclTensorDesc* inputDesc[] = {gradsDesc, featuresDesc, weightsDesc, updatesDesc};
    aclDataBuffer* inputBuffers[] = {gradsBuffer, featuresBuffer, weightsBuffer, updatesBuffer};
    aclTensorDesc* outputDesc[] = {daDesc};
    aclDataBuffer* outputBuffers[] = {daBuffer};

    return aclopCompileAndExecuteFunc("PReluGradReduce", 4, inputDesc, inputBuffers, 1, outputDesc, outputBuffers,
        nullptr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr, stream);
}

std::vector<float> MakeUpdates(const CaseSpec& testCase)
{
    size_t total = static_cast<size_t>(GetShapeSize(testCase.updatesShape));
    std::vector<float> updates(total, 1.0F);
    if (testCase.allOnes) {
        std::fill(updates.begin(), updates.end(), testCase.inputScale);
        return updates;
    }
    for (size_t i = 0; i < total; ++i) {
        if (testCase.inputPattern == InputPattern::SIGNED_PATTERN) {
            float sign = (i % 2 == 0) ? 1.0F : -1.0F;
            updates[i] = sign * static_cast<float>((i % 11) + 1) * testCase.inputScale;
        } else if (testCase.inputPattern == InputPattern::CANCEL_PATTERN) {
            size_t outputCycle = static_cast<size_t>(GetShapeSize(testCase.weightsShape));
            size_t reduceIndex = outputCycle == 0 ? i : i / outputCycle;
            float base = static_cast<float>((i % 7) + 1) * testCase.inputScale;
            updates[i] = (reduceIndex % 2 == 0) ? base : -base;
        } else {
            updates[i] = static_cast<float>((i % 13) + 1) * testCase.inputScale;
        }
    }
    return updates;
}

void DecodeIndex(const std::vector<int64_t>& shape, int64_t index, std::vector<int64_t>& coord)
{
    coord.assign(shape.size(), 0);
    for (int64_t axis = static_cast<int64_t>(shape.size()) - 1; axis >= 0; --axis) {
        coord[axis] = index % shape[axis];
        index = index / shape[axis];
    }
}

std::vector<float> GoldenSharedAll(const std::vector<float>& updates)
{
    float sum = 0.0F;
    for (auto value : updates) {
        sum += value;
    }
    return {sum};
}

std::vector<float> GoldenChannel(const std::vector<float>& updates, const std::vector<int64_t>& shape)
{
    int64_t n = shape[0];
    int64_t c = shape[1];
    int64_t h = shape.size() > 2 ? shape[2] : 1;
    int64_t w = shape.size() > 3 ? shape[3] : 1;
    std::vector<float> golden(static_cast<size_t>(c), 0.0F);
    for (int64_t nIdx = 0; nIdx < n; ++nIdx) {
        for (int64_t cIdx = 0; cIdx < c; ++cIdx) {
            for (int64_t hIdx = 0; hIdx < h; ++hIdx) {
                for (int64_t wIdx = 0; wIdx < w; ++wIdx) {
                    int64_t offset = ((nIdx * c + cIdx) * h + hIdx) * w + wIdx;
                    golden[static_cast<size_t>(cIdx)] += updates[static_cast<size_t>(offset)];
                }
            }
        }
    }
    return golden;
}

std::vector<float> GoldenNc1hwc0(const std::vector<float>& updates, const std::vector<int64_t>& shape)
{
    int64_t n = shape[0];
    int64_t c1 = shape[1];
    int64_t h = shape[2];
    int64_t w = shape[3];
    int64_t c0 = shape[4];
    std::vector<float> golden(static_cast<size_t>(c1 * c0), 0.0F);
    for (int64_t nIdx = 0; nIdx < n; ++nIdx) {
        for (int64_t c1Idx = 0; c1Idx < c1; ++c1Idx) {
            for (int64_t hIdx = 0; hIdx < h; ++hIdx) {
                for (int64_t wIdx = 0; wIdx < w; ++wIdx) {
                    for (int64_t c0Idx = 0; c0Idx < c0; ++c0Idx) {
                        int64_t offset = ((((nIdx * c1 + c1Idx) * h + hIdx) * w + wIdx) * c0 + c0Idx);
                        golden[static_cast<size_t>(c1Idx * c0 + c0Idx)] += updates[static_cast<size_t>(offset)];
                    }
                }
            }
        }
    }
    return golden;
}

std::vector<float> GoldenRank3TailReduceN(const std::vector<float>& updates, const std::vector<int64_t>& shape)
{
    int64_t n = shape[0];
    int64_t c = shape[1];
    int64_t w = shape[2];
    std::vector<float> golden(static_cast<size_t>(c * w), 0.0F);
    for (int64_t nIdx = 0; nIdx < n; ++nIdx) {
        for (int64_t cIdx = 0; cIdx < c; ++cIdx) {
            for (int64_t wIdx = 0; wIdx < w; ++wIdx) {
                int64_t inputOffset = (nIdx * c + cIdx) * w + wIdx;
                int64_t outputOffset = cIdx * w + wIdx;
                golden[static_cast<size_t>(outputOffset)] += updates[static_cast<size_t>(inputOffset)];
            }
        }
    }
    return golden;
}

std::vector<float> GoldenRank4TailReduceN(const std::vector<float>& updates, const std::vector<int64_t>& shape)
{
    int64_t n = shape[0];
    int64_t c = shape[1];
    int64_t h = shape[2];
    int64_t w = shape[3];
    std::vector<float> golden(static_cast<size_t>(c * h * w), 0.0F);
    for (int64_t nIdx = 0; nIdx < n; ++nIdx) {
        for (int64_t cIdx = 0; cIdx < c; ++cIdx) {
            for (int64_t hIdx = 0; hIdx < h; ++hIdx) {
                for (int64_t wIdx = 0; wIdx < w; ++wIdx) {
                    int64_t inputOffset = ((nIdx * c + cIdx) * h + hIdx) * w + wIdx;
                    int64_t outputOffset = (cIdx * h + hIdx) * w + wIdx;
                    golden[static_cast<size_t>(outputOffset)] += updates[static_cast<size_t>(inputOffset)];
                }
            }
        }
    }
    return golden;
}

std::vector<float> GoldenNdWeightShape(const std::vector<float>& updates, const CaseSpec& testCase)
{
    if (testCase.updatesShape.size() == 3 && testCase.weightsShape.size() == 2 &&
        testCase.weightsShape[0] == testCase.updatesShape[1] &&
        testCase.weightsShape[1] == testCase.updatesShape[2]) {
        return GoldenRank3TailReduceN(updates, testCase.updatesShape);
    }

    if (testCase.updatesShape.size() == 4 && testCase.weightsShape.size() == 3 &&
        testCase.weightsShape[0] == testCase.updatesShape[1] &&
        testCase.weightsShape[1] == testCase.updatesShape[2] &&
        testCase.weightsShape[2] == testCase.updatesShape[3]) {
        return GoldenRank4TailReduceN(updates, testCase.updatesShape);
    }

    int64_t outputLength = GetShapeSize(testCase.weightsShape);
    int64_t totalLength = GetShapeSize(testCase.updatesShape);
    std::vector<float> golden(static_cast<size_t>(outputLength), 0.0F);
    std::vector<int64_t> outCoord;
    std::vector<int64_t> inCoord;
    for (int64_t outIndex = 0; outIndex < outputLength; ++outIndex) {
        DecodeIndex(testCase.weightsShape, outIndex, outCoord);
        float sum = 0.0F;
        for (int64_t inIndex = 0; inIndex < totalLength; ++inIndex) {
            DecodeIndex(testCase.updatesShape, inIndex, inCoord);
            bool match = true;
            for (size_t axis = 0; axis < testCase.weightsShape.size(); ++axis) {
                int64_t inputDim = testCase.updatesShape[axis + 1];
                int64_t weightDim = testCase.weightsShape[axis];
                if (inputDim == weightDim) {
                    match = match && (inCoord[axis + 1] == outCoord[axis]);
                } else {
                    match = match && (outCoord[axis] == 0);
                }
            }
            if (match) {
                sum += updates[static_cast<size_t>(inIndex)];
            }
        }
        golden[static_cast<size_t>(outIndex)] = sum;
    }
    return golden;
}

std::vector<float> MakeGolden(const std::vector<float>& updates, const CaseSpec& testCase)
{
    bool sharedAll = testCase.weightsShape.size() == 1 && testCase.weightsShape[0] == 1;
    if (sharedAll) {
        return GoldenSharedAll(updates);
    }
    if (testCase.updatesShape.size() == 5) {
        return GoldenNc1hwc0(updates, testCase.updatesShape);
    }
    if (testCase.weightsShape.size() == testCase.updatesShape.size() - 1 && testCase.weightsShape.size() != 1) {
        return GoldenNdWeightShape(updates, testCase);
    }
    return GoldenChannel(updates, testCase.updatesShape);
}

template <typename T>
std::vector<T> CastVector(const std::vector<float>& values)
{
    std::vector<T> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = static_cast<T>(values[i]);
    }
    return result;
}

template <>
std::vector<uint16_t> CastVector<uint16_t>(const std::vector<float>& values)
{
    std::vector<uint16_t> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = FloatToHalf(values[i]);
    }
    return result;
}

std::vector<uint16_t> CastVectorBf16(const std::vector<float>& values)
{
    std::vector<uint16_t> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = FloatToBFloat16(values[i]);
    }
    return result;
}

template <typename T>
std::vector<float> ToFloatVector(const std::vector<T>& values)
{
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = static_cast<float>(values[i]);
    }
    return result;
}

template <>
std::vector<float> ToFloatVector<uint16_t>(const std::vector<uint16_t>& values)
{
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = HalfToFloat(values[i]);
    }
    return result;
}

std::vector<float> ToFloatVectorBf16(const std::vector<uint16_t>& values)
{
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = BFloat16ToFloat(values[i]);
    }
    return result;
}

template <typename T>
std::vector<float> OutputToFloatVector(const std::vector<T>& values, bool isBf16)
{
    (void)isBf16;
    return ToFloatVector(values);
}

std::vector<float> OutputToFloatVector(const std::vector<uint16_t>& values, bool isBf16)
{
    return isBf16 ? ToFloatVectorBf16(values) : ToFloatVector(values);
}

void PrintShape(const std::vector<int64_t>& shape)
{
    LOG_PRINT("[");
    for (size_t i = 0; i < shape.size(); ++i) {
        LOG_PRINT("%ld%s", shape[i], i + 1 == shape.size() ? "" : ",");
    }
    LOG_PRINT("]");
}

void PrintSample(const std::vector<float>& expected, const std::vector<float>& actual)
{
    size_t printSize = std::min(kSamplePrintSize, expected.size());
    LOG_PRINT("sample expected/actual:");
    for (size_t i = 0; i < printSize; ++i) {
        LOG_PRINT(" %.6f/%.6f", expected[i], actual[i]);
    }
    LOG_PRINT("\n");
}

template <typename T>
int RunTypedCase(const CaseSpec& testCase, aclrtStream stream, const std::vector<T>& updatesHostData,
    const std::vector<T>& expectedHostData, float tolerance, bool isBf16 = false)
{
    size_t updatesLength = static_cast<size_t>(GetShapeSize(testCase.updatesShape));
    size_t outputLength = static_cast<size_t>(GetShapeSize(testCase.weightsShape));
    std::vector<T> gradsHostData(updatesLength, static_cast<T>(0));
    std::vector<T> featuresHostData(updatesLength, static_cast<T>(0));
    std::vector<T> weightsHostData(outputLength, static_cast<T>(1));
    std::vector<T> daHostData(outputLength, static_cast<T>(0));

    void* gradsDeviceAddr = nullptr;
    void* featuresDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* updatesDeviceAddr = nullptr;
    void* daDeviceAddr = nullptr;
    aclDataBuffer* gradsBuffer = nullptr;
    aclDataBuffer* featuresBuffer = nullptr;
    aclDataBuffer* weightsBuffer = nullptr;
    aclDataBuffer* updatesBuffer = nullptr;
    aclDataBuffer* daBuffer = nullptr;

    auto ret = CreateDeviceBuffer(gradsHostData, &gradsDeviceAddr, &gradsBuffer);
    std::unique_ptr<aclDataBuffer, aclError (*)(aclDataBuffer*)> gradsBufferPtr(gradsBuffer, DestroyDataBuffer);
    std::unique_ptr<void, aclError (*)(void*)> gradsDeviceAddrPtr(gradsDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateDeviceBuffer(featuresHostData, &featuresDeviceAddr, &featuresBuffer);
    std::unique_ptr<aclDataBuffer, aclError (*)(aclDataBuffer*)> featuresBufferPtr(featuresBuffer,
        DestroyDataBuffer);
    std::unique_ptr<void, aclError (*)(void*)> featuresDeviceAddrPtr(featuresDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateDeviceBuffer(weightsHostData, &weightsDeviceAddr, &weightsBuffer);
    std::unique_ptr<aclDataBuffer, aclError (*)(aclDataBuffer*)> weightsBufferPtr(weightsBuffer,
        DestroyDataBuffer);
    std::unique_ptr<void, aclError (*)(void*)> weightsDeviceAddrPtr(weightsDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateDeviceBuffer(updatesHostData, &updatesDeviceAddr, &updatesBuffer);
    std::unique_ptr<aclDataBuffer, aclError (*)(aclDataBuffer*)> updatesBufferPtr(updatesBuffer,
        DestroyDataBuffer);
    std::unique_ptr<void, aclError (*)(void*)> updatesDeviceAddrPtr(updatesDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateDeviceBuffer(daHostData, &daDeviceAddr, &daBuffer);
    std::unique_ptr<aclDataBuffer, aclError (*)(aclDataBuffer*)> daBufferPtr(daBuffer, DestroyDataBuffer);
    std::unique_ptr<void, aclError (*)(void*)> daDeviceAddrPtr(daDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 【改成 Aclnn 接口】
    ret = RunPReluGradReduceByAclnn(gradsBuffer, featuresBuffer, weightsBuffer, updatesBuffer, daBuffer, testCase,
        stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnn PReluGradReduce failed, case=%s ERROR: %d\n", testCase.name, ret);
        return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<T> actualHostData(outputLength, static_cast<T>(0));
    ret = aclrtMemcpy(actualHostData.data(), outputLength * sizeof(T), daDeviceAddr, outputLength * sizeof(T),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy da from device to host failed. ERROR: %d\n", ret); return ret);

    std::vector<float> expectedFloat = OutputToFloatVector(expectedHostData, isBf16);
    std::vector<float> actualFloat = OutputToFloatVector(actualHostData, isBf16);
    float maxDiff = 0.0F;
    float maxRelDiff = 0.0F;
    for (size_t i = 0; i < outputLength; ++i) {
        float diff = std::fabs(expectedFloat[i] - actualFloat[i]);
        float denom = std::max(std::fabs(expectedFloat[i]), 1.0F);
        maxDiff = std::max(maxDiff, diff);
        maxRelDiff = std::max(maxRelDiff, diff / denom);
    }

    LOG_PRINT("[CASE] %s updates=", testCase.name);
    PrintShape(testCase.updatesShape);
    LOG_PRINT(" weights=");
    PrintShape(testCase.weightsShape);
    LOG_PRINT(" output=%zu max_diff=%.9f max_relative_diff=%.9f\n", outputLength, maxDiff, maxRelDiff);
    PrintSample(expectedFloat, actualFloat);
    float relTolerance = isBf16 || std::is_same<T, uint16_t>::value ? kFp16RelTolerance : kFp32RelTolerance;
    CHECK_RET(maxDiff <= tolerance || maxRelDiff <= relTolerance,
        LOG_PRINT("[FAILED] %s\n", testCase.name); return 1);

    if (IsBenchmarkEnabled() && testCase.benchmark) {
        auto wallStart = std::chrono::steady_clock::now();
        uint32_t warmup = GetEnvUint32("PRELU_GRAD_REDUCE_BENCH_WARMUP", 20);
        uint32_t iterations = GetEnvUint32("PRELU_GRAD_REDUCE_BENCH_ITERS", 1000);
        for (uint32_t i = 0; i < warmup; ++i) {
            ret = RunPReluGradReduceByAclop(gradsBuffer, featuresBuffer, weightsBuffer, updatesBuffer, daBuffer,
                testCase, stream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclopCompileAndExecute warmup failed. ERROR: %d\n", ret);
                return ret);
        }
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("warmup synchronize failed. ERROR: %d\n", ret); return ret);

        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        ret = aclrtCreateEvent(&start);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateEvent start failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateEvent(&end);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateEvent end failed. ERROR: %d\n", ret); return ret);

        ret = aclrtRecordEvent(start, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtRecordEvent start failed. ERROR: %d\n", ret); return ret);
        for (uint32_t i = 0; i < iterations; ++i) {
            ret = RunPReluGradReduceByAclop(gradsBuffer, featuresBuffer, weightsBuffer, updatesBuffer, daBuffer,
                testCase, stream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclopCompileAndExecute benchmark failed. ERROR: %d\n", ret);
                return ret);
        }
        ret = aclrtRecordEvent(end, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtRecordEvent end failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSynchronizeEvent(end);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeEvent failed. ERROR: %d\n", ret); return ret);

        float elapsedMs = 0.0F;
        ret = aclrtEventElapsedTime(&elapsedMs, start, end);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtEventElapsedTime failed. ERROR: %d\n", ret); return ret);
        double avgUs = static_cast<double>(elapsedMs) * 1000.0 / static_cast<double>(iterations);
        auto wallEnd = std::chrono::steady_clock::now();
        double wallTotalMs = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();
        LOG_PRINT("[ACL-BENCH] case=%s warmup=%u iters=%u total_ms=%.6f avg_us=%.6f wall_total_ms=%.6f\n",
            testCase.name, warmup, iterations, elapsedMs, avgUs, wallTotalMs);

        aclrtDestroyEvent(start);
        aclrtDestroyEvent(end);
    }

    LOG_PRINT("[SUCCESS] %s\n", testCase.name);
    return ACL_SUCCESS;
}

int RunCase(const CaseSpec& testCase, aclrtStream stream)
{
    LOG_PRINT("[RUN] %s\n", testCase.name);
    std::vector<float> updatesFloat = MakeUpdates(testCase);
    std::vector<float> goldenFloat = MakeGolden(updatesFloat, testCase);
    if (testCase.dataType == ACL_FLOAT) {
        return RunTypedCase<float>(testCase, stream, CastVector<float>(updatesFloat), goldenFloat, kFp32Tolerance);
    }
    if (testCase.dataType == ACL_BF16) {
        return RunTypedCase<uint16_t>(testCase, stream, CastVectorBf16(updatesFloat),
            CastVectorBf16(goldenFloat), kFp16Tolerance, true);
    }
    return RunTypedCase<uint16_t>(testCase, stream, CastVector<uint16_t>(updatesFloat),
        CastVector<uint16_t>(goldenFloat), kFp16Tolerance);
}

int PReluGradReduceTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<CaseSpec> cases = {
        {"nd_2d_fp32", {32, 1024}, {1024}, ACL_FLOAT, ACL_FORMAT_ND,
            ACL_FORMAT_ND, ACL_FORMAT_ND, false, false, 1.0F,
            "2D FP32 ND format, reduce axis [0]."},
        {"shared_large_fp32", {32, 10, 32, 128}, {1}, ACL_FLOAT, ACL_FORMAT_ND,
            ACL_FORMAT_ND, ACL_FORMAT_ND, false, true, 1.0F,
            "Shared all-reduce large FP32."},
        {"shared_large_fp16", {32, 10, 32, 128}, {1}, ACL_FLOAT16, ACL_FORMAT_ND,
            ACL_FORMAT_ND, ACL_FORMAT_ND, false, true, 0.001F,
            "Shared all-reduce large FP16."},
        {"shared_large_bf16", {32, 10, 32, 128}, {1}, ACL_BF16, ACL_FORMAT_ND,
            ACL_FORMAT_ND, ACL_FORMAT_ND, false, true, 0.001F,
            "Shared all-reduce large BF16."},
        {"nchw_channel_large_fp32", {32, 10, 32, 128}, {10}, ACL_FLOAT, ACL_FORMAT_ND,
            ACL_FORMAT_ND, ACL_FORMAT_ND, false, true, 1.0F,
            "NCHW channel reduce large FP32."},
        {"nchw_channel_large_fp16", {32, 10, 32, 128}, {10}, ACL_FLOAT16, ACL_FORMAT_ND,
            ACL_FORMAT_ND, ACL_FORMAT_ND, false, true, 0.01F,
            "NCHW channel reduce large FP16."},
        {"nchw_channel_large_bf16", {32, 10, 32, 128}, {10}, ACL_BF16, ACL_FORMAT_ND,
            ACL_FORMAT_ND, ACL_FORMAT_ND, false, true, 0.01F,
            "NCHW channel reduce large BF16."},
        {"nd_rank3_weight_shape_fp32", {20, 32, 125}, {32, 125}, ACL_FLOAT, ACL_FORMAT_ND,
            ACL_FORMAT_ND, ACL_FORMAT_ND, false, true, 1.0F,
            "ND rank3 weight-shape FP32."},
        {"nd_rank3_weight_shape_fp16", {20, 32, 125}, {32, 125}, ACL_FLOAT16, ACL_FORMAT_ND,
            ACL_FORMAT_ND, ACL_FORMAT_ND, false, true, 0.01F,
            "ND rank3 weight-shape FP16."},
        {"nd_rank3_weight_shape_bf16", {20, 32, 125}, {32, 125}, ACL_BF16, ACL_FORMAT_ND,
            ACL_FORMAT_ND, ACL_FORMAT_ND, false, true, 0.01F,
            "ND rank3 weight-shape BF16."},
        {"nc1hwc0_large_fp32", {32, 2, 32, 128, 16}, {2, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NC1HWC0,
            ACL_FORMAT_NC1HWC0, ACL_FORMAT_NC1HWC0, false, true, 1.0F,
            "NC1HWC0 large FP32."},
        {"nc1hwc0_large_fp16", {32, 2, 32, 128, 16}, {2, 1, 1, 16}, ACL_FLOAT16, ACL_FORMAT_NC1HWC0,
            ACL_FORMAT_NC1HWC0, ACL_FORMAT_NC1HWC0, false, true, 0.01F,
            "NC1HWC0 large FP16."},
        {"nc1hwc0_large_bf16", {32, 2, 32, 128, 16}, {2, 1, 1, 16}, ACL_BF16, ACL_FORMAT_NC1HWC0,
            ACL_FORMAT_NC1HWC0, ACL_FORMAT_NC1HWC0, false, true, 0.01F,
            "NC1HWC0 large BF16."},
    };
    
    const char* onlyCase = std::getenv("PRELU_GRAD_REDUCE_CASE");
    uint32_t failedCount = 0;
    uint32_t runCount = 0;
    for (const auto& testCase : cases) {
        if (onlyCase != nullptr && onlyCase[0] != '\0' && std::string(onlyCase) != testCase.name) {
            continue;
        }
        ++runCount;
        ret = RunCase(testCase, stream);
        if (ret != ACL_SUCCESS) {
            ++failedCount;
        }
    }

    CHECK_RET(failedCount == 0, LOG_PRINT("PReluGradReduce multi-case test failed, failed_count=%u\n", failedCount);
        return 1);
    CHECK_RET(runCount != 0, LOG_PRINT("PReluGradReduce no case selected, PRELU_GRAD_REDUCE_CASE=%s\n",
                               onlyCase == nullptr ? "" : onlyCase);
                  return 1);
    LOG_PRINT("PReluGradReduce smoke test pass, case_count=%zu run_count=%u\n", cases.size(),
        runCount);
    LOG_PRINT("[SMOKE] case_count=%zu run_count=%u dtypes=fp32,fp16,bf16 formats=ND,NC1HWC0 "
              "axis_generalization=rank2,rank3,rank4 shared=enabled "
              "large_cases=enabled nc1hwc0_cases=enabled\n",
        cases.size(), runCount);
    return ACL_SUCCESS;
}

int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    LOG_PRINT("[START] test_aclnn_p_relu_grad_reduce\n");

    int32_t deviceId = 0;
    aclrtStream stream = nullptr;

    auto ret = PReluGradReduceTest(deviceId, stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("PReluGradReduceTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
