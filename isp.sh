#!/usr/bin/env bash
set -euo pipefail

ISP_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

#if command -v conda >/dev/null 2>&1; then
#  # Método recomendado por conda >=4.6
#  eval "$(conda shell.bash hook)"
#elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#  source "$HOME/miniconda3/etc/profile.d/conda.sh"
#elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#  source "$HOME/anaconda3/etc/profile.d/conda.sh"
#else
#  echo "No installed Miniconda/Anaconda." >&2
#  exit 1
#fi

source activate isp

# Detectar SO (solo por si lo usas en otra parte)
if [[ "$(uname -s)" == "Darwin" ]]; then
  export OS="MacOSX"
else
  export OS="Linux"
fi

# Wayland (no aplica en macOS, pero lo dejamos)
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]]; then
  export QT_QPA_PLATFORM=wayland
fi

# Activar entorno (usa sintaxis nueva)
conda activate isp

pushd "${ISP_DIR}" > /dev/null
python start_isp.py
popd > /dev/null
