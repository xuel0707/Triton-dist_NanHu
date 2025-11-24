export CUDA_LAUNCH_BLOCKING=0
export TORCH_CPP_LOG_LEVEL=1
export NCCL_DEBUG=ERROR

SCRIPT_DIR="$(pwd)"
SCRIPT_DIR=$(realpath ${SCRIPT_DIR})

# 1. Check if NVSHMEM_HOME environment variable is set
if [ -n "$NVSHMEM_HOME" ]; then
  echo "Found NVSHMEM_HOME from environment variable: $NVSHMEM_HOME"
else
  # 2. Try to find from Python command
  NVSHMEM_HOME=$(python -c "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))" 2>/dev/null)

  if [ -n "$NVSHMEM_HOME" ]; then
    echo "Found NVSHMEM_HOME from Python nvidia-nvshmem-cu12: $NVSHMEM_HOME"
  else
    # 3. Fallback to ldconfig
    NVSHMEM_HOME=$(ldconfig -p | grep 'libnvshmem_host' | awk '{print $NF}' | xargs dirname | head -n 1)

    if [ -n "$NVSHMEM_HOME" ]; then
      echo "Found NVSHMEM_HOME from ldconfig: $NVSHMEM_HOME"
    else
      echo "warning: NVSHMEM_HOME could not be determined."
    fi
  fi
fi


OMPI_BUILD=${SCRIPT_DIR}/shmem/rocshmem_bind/ompi_build/install/ompi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVSHMEM_HOME}/lib:${OMPI_BUILD}/lib
export NVSHMEM_DISABLE_CUDA_VMM=1 # moving from cpp to shell
export NVSHMEM_BOOTSTRAP=UID

export TRITON_CACHE_DIR=${SCRIPT_DIR}/triton_cache

export PYTHONPATH=$PYTHONPATH:${SCRIPT_DIR}/python
mkdir -p ${SCRIPT_DIR}/triton_cache

if ! command -v ifconfig &> /dev/null; then
    echo "ifconfig is not available. Installing net-tools..."
    
    apt update && apt install -y net-tools
    
    if ! command -v ifconfig &> /dev/null; then
        echo "Failed to install or run ifconfig. Please check your system."
        exit 1
    fi
    echo "net-tools installed successfully."
fi

NET_IFACE=$(ifconfig | grep '^[a-zA-Z]' | awk '{print $1}' | sed 's/://' \
  | grep -vE '^(lo|docker0)$' \
  | while read iface; do
      if ifconfig "$iface" 2>/dev/null | grep -E 'inet6 ' | grep -v '127.0.0.1' > /dev/null; then
          echo "$iface"
          break
      fi
    done)

if [ -z "$NET_IFACE" ]; then
    echo "No valid network interface with IP found."
    exit 1
fi

echo "Using network interface: $NET_IFACE"

# set Network interface of Gloo settings for distributed training by CPU-based collective communication
export GLOO_SOCKET_IFNAME=$NET_IFACE

# set Network interface of NCCL settings for distributed training by GPU Communication
export NCCL_SOCKET_IFNAME=$NET_IFACE

# The network interface used by NVSHMEM for inter-node discovery during startup; 
# it must be a low-latency, high-bandwidth IB or RoCE network interface.
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=$NET_IFACE

# export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-1000000000}
export NVSHMEM_SYMMETRIC_SIZE=20g 
echo "Environment variables set:"
echo "  GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
echo "  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "  NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=$NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"

