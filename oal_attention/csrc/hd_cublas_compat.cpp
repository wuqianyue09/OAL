#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/library.h>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace {

enum class DenseLayout { RowMajor, SimpleTranspose };

void check_cublas(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      operation,
      " failed with cuBLAS status ",
      static_cast<int>(status));
}

DenseLayout classify_dense_layout(
    const at::Tensor& tensor,
    int64_t rows,
    int64_t columns,
    const char* name) {
  const bool row_major = tensor.stride(2) == 1 &&
      tensor.stride(1) == columns &&
      tensor.stride(0) == rows * columns;
  const bool simple_transpose = tensor.stride(1) == 1 &&
      tensor.stride(2) == rows &&
      tensor.stride(0) == rows * columns;
  TORCH_CHECK(
      row_major || simple_transpose,
      name,
      " must use exact dense row-major or simple-transpose layout");
  return row_major ? DenseLayout::RowMajor : DenseLayout::SimpleTranspose;
}

void validate_storage_bounds(
    const at::Tensor& tensor,
    int64_t matrix_elements,
    const char* name) {
  TORCH_CHECK(tensor.storage_offset() >= 0, name, " storage_offset is negative");
  const auto batch = tensor.size(0);
  const auto last_element = tensor.storage_offset() +
      (batch - 1) * tensor.stride(0) + matrix_elements;
  const auto storage_elements =
      static_cast<int64_t>(tensor.storage().nbytes() / tensor.element_size());
  TORCH_CHECK(
      last_element <= storage_elements,
      name,
      " dense view exceeds its storage bounds");
}

class PointerModeGuard {
 public:
  explicit PointerModeGuard(cublasHandle_t handle) : handle_(handle) {
    check_cublas(
        cublasGetPointerMode(handle_, &original_),
        "cublasGetPointerMode");
    if (original_ != CUBLAS_POINTER_MODE_HOST) {
      check_cublas(
          cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
          "cublasSetPointerMode(host)");
      changed_ = true;
    }
  }

  PointerModeGuard(const PointerModeGuard&) = delete;
  PointerModeGuard& operator=(const PointerModeGuard&) = delete;

  ~PointerModeGuard() {
    if (changed_) {
      (void)cublasSetPointerMode(handle_, original_);
    }
  }

 private:
  cublasHandle_t handle_{};
  cublasPointerMode_t original_{CUBLAS_POINTER_MODE_HOST};
  bool changed_{false};
};

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(
      value > 0 && value <= std::numeric_limits<int>::max(),
      name,
      " must be positive and fit the cuBLAS int interface");
  return static_cast<int>(value);
}

void bmm_fp32(
    const at::Tensor& left,
    const at::Tensor& right,
    at::Tensor& out) {
  TORCH_CHECK(left.is_cuda(), "left must be a CUDA tensor");
  TORCH_CHECK(right.is_cuda(), "right must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(
      left.device() == right.device() && left.device() == out.device(),
      "left, right and out must use the same CUDA device");
  TORCH_CHECK(left.layout() == c10::kStrided, "left must be strided");
  TORCH_CHECK(right.layout() == c10::kStrided, "right must be strided");
  TORCH_CHECK(out.layout() == c10::kStrided, "out must be strided");
  TORCH_CHECK(left.dim() == 3, "left must be rank 3");
  TORCH_CHECK(right.dim() == 3, "right must be rank 3");
  TORCH_CHECK(out.dim() == 3, "out must be rank 3");
  TORCH_CHECK(left.scalar_type() == at::kBFloat16, "left must be bfloat16");
  TORCH_CHECK(right.scalar_type() == at::kBFloat16, "right must be bfloat16");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");

  const auto batch = left.size(0);
  const auto rows = left.size(1);
  const auto reduction = left.size(2);
  const auto columns = right.size(2);
  TORCH_CHECK(
      batch > 0 && rows > 0 && reduction > 0 && columns > 0,
      "zero-size contractions are unsupported");
  TORCH_CHECK(
      right.size(0) == batch && right.size(1) == reduction,
      "right shape does not match left");
  TORCH_CHECK(
      out.size(0) == batch && out.size(1) == rows &&
          out.size(2) == columns,
      "out shape does not match the contraction");

  const auto left_layout =
      classify_dense_layout(left, rows, reduction, "left");
  const auto right_layout =
      classify_dense_layout(right, reduction, columns, "right");
  TORCH_CHECK(
      out.stride(2) == 1 && out.stride(1) == columns &&
          out.stride(0) == rows * columns,
      "out must use exact dense row-major layout");
  validate_storage_bounds(left, rows * reduction, "left");
  validate_storage_bounds(right, reduction * columns, "right");
  validate_storage_bounds(out, rows * columns, "out");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, left);
  at::assert_no_overlap(out, right);

  const int batch_i = checked_int(batch, "batch");
  const int m = checked_int(columns, "columns");
  const int n = checked_int(rows, "rows");
  const int k = checked_int(reduction, "reduction");
  const int lda = right_layout == DenseLayout::RowMajor ? m : k;
  const int ldb = left_layout == DenseLayout::RowMajor ? k : n;
  const int ldc = m;
  const auto op_a = right_layout == DenseLayout::RowMajor ? CUBLAS_OP_N
                                                           : CUBLAS_OP_T;
  const auto op_b = left_layout == DenseLayout::RowMajor ? CUBLAS_OP_N
                                                          : CUBLAS_OP_T;
  const long long stride_a = reduction * columns;
  const long long stride_b = rows * reduction;
  const long long stride_c = rows * columns;

  c10::cuda::CUDAGuard device_guard(left.device());
  const auto current_stream = c10::cuda::getCurrentCUDAStream(left.get_device());
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t handle_stream = nullptr;
  check_cublas(cublasGetStream(handle, &handle_stream), "cublasGetStream");
  TORCH_CHECK(
      handle_stream == static_cast<cudaStream_t>(current_stream),
      "PyTorch cuBLAS handle is not bound to the current CUDA stream");

  PointerModeGuard pointer_mode_guard(handle);
  float alpha = 1.0f;
  float beta = 0.0f;
  const auto status = cublasGemmStridedBatchedEx(
      handle,
      op_a,
      op_b,
      m,
      n,
      k,
      &alpha,
      right.data_ptr(),
      CUDA_R_16BF,
      lda,
      stride_a,
      left.data_ptr(),
      CUDA_R_16BF,
      ldb,
      stride_b,
      &beta,
      out.data_ptr(),
      CUDA_R_32F,
      ldc,
      stride_c,
      batch_i,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
  check_cublas(status, "cublasGemmStridedBatchedEx");
  torch::autograd::increment_version(out);
}

std::string escape_json(const char* value) {
  std::ostringstream escaped;
  for (const char* cursor = value; cursor != nullptr && *cursor != '\0'; ++cursor) {
    if (*cursor == '\\' || *cursor == '"') {
      escaped << '\\';
    }
    escaped << *cursor;
  }
  return escaped.str();
}

std::string runtime_metadata() {
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  int runtime_version = 0;
  cublasMath_t math_mode{};
  cublasPointerMode_t pointer_mode{};
  check_cublas(cublasGetVersion(handle, &runtime_version), "cublasGetVersion");
  check_cublas(cublasGetMathMode(handle, &math_mode), "cublasGetMathMode");
  check_cublas(
      cublasGetPointerMode(handle, &pointer_mode),
      "cublasGetPointerMode");
  Dl_info provider_info{};
  const auto symbol = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(&cublasGetVersion));
  TORCH_CHECK(dladdr(symbol, &provider_info) != 0, "dladdr failed for cuBLAS");
  TORCH_CHECK(provider_info.dli_fname != nullptr, "cuBLAS provider path is missing");

  std::ostringstream result;
  result << "{\"compile_cuda_version\":" << CUDART_VERSION
         << ",\"runtime_cublas_version\":" << runtime_version
         << ",\"math_mode\":" << static_cast<int>(math_mode)
         << ",\"pointer_mode\":" << static_cast<int>(pointer_mode)
         << ",\"provider_realpath\":\""
         << escape_json(provider_info.dli_fname) << "\"}";
  return result.str();
}

}  // namespace

TORCH_LIBRARY(hd_cublas_compat, library) {
  library.def("bmm_fp32(Tensor left, Tensor right, *, Tensor(a!) out) -> ()");
  library.def("runtime_metadata() -> str", TORCH_FN(runtime_metadata));
}

TORCH_LIBRARY_IMPL(hd_cublas_compat, CUDA, library) {
  library.impl("bmm_fp32", TORCH_FN(bmm_fp32));
}
