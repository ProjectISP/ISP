#!/usr/bin/env bash
set -euo pipefail

# Dir del script
ISP_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 && pwd -P )"

# Asegura PS1 para evitar problemas con set -u en hooks de conda
: "${PS1:=}"

# Detectar OS (por si lo usas en otra parte)
if [[ "$(uname -s)" == "Darwin" ]]; then
  export OS="MacOSX"
else
  export OS="Linux"
fi

# Wayland (no aplica en macOS; solo si no está ya seteado)
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=wayland
fi

# Localiza conda o micromamba
find_conda_sh() {
  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "$HOME/miniforge3/etc/profile.d/conda.sh"
    "$HOME/mambaforge/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "/opt/anaconda3/etc/profile.d/conda.sh"
  )
  for f in "${candidates[@]}"; do
    [[ -f "$f" ]] && { echo "$f"; return 0; }
  done
  return 1
}

# Preferimos conda RUN para evitar activar el entorno
if command -v conda >/dev/null 2>&1; then
  CONDA_CMD="conda"
elif CONDA_SH="$(find_conda_sh)"; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  # Tras source, debería existir 'conda' en PATH
  if ! command -v conda >/dev/null 2>&1; then
    echo "No se pudo cargar conda desde: $CONDA_SH" >&2
    exit 1
  fi
  CONDA_CMD="conda"
elif command -v micromamba >/dev/null 2>&1; then
  CONDA_CMD="micromamba"
else
  echo "No encuentro conda/miniforge/mambaforge ni micromamba en este sistema." >&2
  exit 1
fi

# Comprueba que el entorno 'isp' existe
if ! "$CONDA_CMD" env list | grep -E '(^|\s)isp(\s|$)' >/dev/null 2>&1; then
  echo "El entorno 'isp' no existe. Crea uno (p.ej. 'conda env create -n isp -f environment.yml' o 'conda create -n isp python')." >&2
  exit 1
fi

# Ejecuta sin activar el entorno (más portable)
cd "$ISP_DIR"
if [[ "$CONDA_CMD" == "conda" ]]; then
  exec conda run --no-capture-output -n isp python -u start_isp.py
else
  # micromamba
  exec micromamba run -n isp python -u start_isp.py
fi
