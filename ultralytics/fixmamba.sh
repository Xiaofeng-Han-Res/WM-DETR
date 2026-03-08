#!/usr/bin/env bash
# fix_mamba_glibc.sh
# 自动检测 glibc 版本，决定是源码编译 mamba-ssm 还是安装 wheel

set -euo pipefail
PY="${PYTHON:-python}"

echo "==> 检测 glibc 版本..."
GLIBC_VER=$(ldd --version 2>/dev/null | head -n1 | sed -E 's/.* ([0-9]+\.[0-9]+).*/\1/')
echo "当前 glibc: $GLIBC_VER"

needs_source_build=0
awk "BEGIN{exit !($GLIBC_VER < 2.32)}" || needs_source_build=1

# 确保 numpy 版本合适
$PY -m pip install -U --no-cache-dir "numpy<2"

if [[ $needs_source_build -eq 1 ]]; then
  echo "==> glibc < 2.32，强制源码编译 mamba-ssm"
  pip uninstall -y mamba-ssm || true
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"7.0;7.5;8.0;8.6;8.9"}
  export MAX_JOBS=$(nproc)
  $PY -m pip install --no-binary :all: --no-cache-dir -v mamba-ssm
else
  echo "==> glibc >= 2.32，尝试安装预编译 wheel"
  $PY -m pip install -U --no-cache-dir mamba-ssm
fi

echo "==> 测试 selective_scan 导入..."
$PY - <<'PY'
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("✅ selective_scan 成功导入")
except Exception as e:
    print("❌ selective_scan 导入失败:", e)
    print("尝试 Triton 回退...")
    import os
    os.environ["SELECTIVE_SCAN_DISABLE_CUDA"] = "1"
    os.environ["MAMBA_FORCE_TRITON"] = "1"
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("⚠️ 已回退到 Triton/CPU 实现")
PY

echo "==> 完成"
