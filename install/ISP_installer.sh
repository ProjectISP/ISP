#!/bin/bash

# ******************************************************************************
# *   ISP: Integrated Seismic Program                                          *
# *                                                                            *
# *   A Python GUI for earthquake seismology and seismic signal processing     *
# *                                                                            *
# *   Copyright (C) 2025     Roberto Cabieces                                  *
# *   Copyright (C) 2025     Andrés Olivar-Castaño                              *
# *   Copyright (C) 2025     Thiago C. Junqueira                               *
# *   Copyright (C) 2025     Jesús Relinque                                    *
# *                                                                            *
# *   This file is part of ISP.                                                *
# *                                                                            *
# *   ISP is free software: you can redistribute it and/or modify it under the *
# *   terms of the GNU Lesser General Public License (LGPL) as published by    *
# *   the Free Software Foundation, either version 3 of the License, or (at    *
# *   your option) any later version.                                          *
# *                                                                            *
# *   ISP is distributed in the hope that it will be useful for any user or,   *
# *   institution, but WITHOUT ANY WARRANTY; without even the implied warranty *
# *   of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU      *
# *   Lesser General Public License (LGPL) for more details.                   *
# *                                                                            *
# *   You should have received a copy of the GNU LGPL license along with       *
# *   ISP. If not, see <http://www.gnu.org/licenses/>.                         *
# ******************************************************************************

# ******************************************************************************
# * INTEGRATED SEISMIC PROGRAM INSTALLER USING CONDA ENVIRONMENT               *
# ******************************************************************************

set -euo pipefail

# --- System deps check (new): run BEFORE any conda steps ----------------------
ensure_system_deps() {
  local os="$(uname -s)"
  if [[ "$os" != "Linux" ]]; then
    echo "[deps] Non-Linux system detected ($os). Skipping system package checks."
    return 0
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[deps] Non-Debian/Ubuntu Linux (no apt-get found). Skipping system package checks."
    return 0
  fi

  # Determine sudo usage
  local SUDO=""
  if [[ $EUID -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      SUDO="sudo"
    else
      echo "[deps] Need root privileges but 'sudo' not found. Please run this script as root."
      exit 1
    fi
  fi

  # Packages to check/install
  local base_pkgs=(build-essential pkg-config xdg-utils libpulse-mainloop-glib0)
  local xcb_pkgs=(libxcb-xinerama0 libxcb1 libx11-xcb1 libxext6 libxkbcommon-x11-0)

  # Find missing packages
  local missing=()
  for p in "${base_pkgs[@]}" "${xcb_pkgs[@]}"; do
    if ! dpkg -s "$p" >/dev/null 2>&1; then
      missing+=("$p")
    fi
  done

  if ((${#missing[@]})); then
    echo "[deps] Missing system packages detected: ${missing[*]}"
    read -r -p "Install missing system packages now? [Y/n] " resp
    resp="${resp:-Y}"
    case "${resp,,}" in
      y|yes)
        echo "[deps] Proceeding with installation (Y)."
        export DEBIAN_FRONTEND=noninteractive
        $SUDO apt-get update
        $SUDO apt-get install -y "${missing[@]}"
        ;;
      n|no)
        echo "[deps] Skipping installation (N). Some components may fail later."
        ;;
      *)
        echo "[deps] Unrecognized response. Skipping installation."
        ;;
    esac
  else
    echo "[deps] All required system packages already installed."
  fi

  # Offer to force-reinstall XCB libs (helps when libs are present but broken)
  read -r -p "Reinstall low-level XCB libraries to fix potential GUI issues? [y/N] " re_resp
  re_resp="${re_resp:-N}"
  if [[ "${re_resp,,}" =~ ^y ]]; then
    echo "[deps] Reinstalling XCB libraries (Y)."
    export DEBIAN_FRONTEND=noninteractive
    $SUDO apt-get update
    $SUDO apt-get install --reinstall -y libxcb-xinerama0
    $SUDO apt-get install --reinstall -y libxcb1 libx11-xcb1 libxext6 libxkbcommon-x11-0
  else
    echo "[deps] Skipping XCB libraries reinstall (N)."
  fi
}

ensure_system_deps
# --- End System deps check (new) ---------------------------------------------

# Ensure conda is available and load shell functions (activate/conda run)
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Please install Miniconda/Anaconda and ensure 'conda' is on PATH."
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Function to detect OS and create Conda environment
create_isp_env() {
  OS_TYPE=$(uname -s)
  echo "Operating System detected: $OS_TYPE"

  if [[ $OS_TYPE == "Darwin" ]]; then
    echo "MacOS detected. Checking processor type..."
    CPU_INFO=$(sysctl -n machdep.cpu.brand_string)
    echo "Processor Info: $CPU_INFO"

    if [[ $CPU_INFO == *"Apple"* ]]; then
      echo "Apple Silicon (M1/M2) detected."
      export OS="MacOSX-ARM"
      ENV_FILE="./mac_installer/mac_environment_arm.yml"
    else
      echo "Intel processor detected."
      export OS="MacOSX-Intel"
      ENV_FILE="./mac_installer/mac_environment.yml"
    fi
  elif [[ $OS_TYPE == "Linux" ]]; then
    echo "Linux detected."
    export OS="Linux"
    ENV_FILE="./linux_installer/linux_environment.yml"
  else
    echo "Unsupported operating system: $OS_TYPE"
    exit 1
  fi

  echo "Using Conda environment file: $ENV_FILE"
  conda env create -f "$ENV_FILE"
}

ISP_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/.."

# Check if the ISP environment exists
if conda env list | grep -q '^isp\s'; then
  echo "'isp' environment already exists."
  read -p "Do you want to remove the existing environment and reinstall ISP? (Y/N): " CHOICE
  CHOICE=$(echo "$CHOICE" | tr '[:upper:]' '[:lower:]')
  if [[ "$CHOICE" == "y" ]]; then
    echo "Removing existing 'isp' environment..."
    conda remove --name isp --all -y
    echo "Reinstalling ISP environment..."
    create_isp_env
  else
    echo "Skipping environment reinstallation. Proceeding with the existing setup."
  fi
else
  echo "No 'isp' environment found. Proceeding to create one."
  create_isp_env
fi

echo "ISP environment process finished"

# Compile packages
pushd ${ISP_DIR} > /dev/null

# --- Key change: use conda run instead of source activate ---
conda deactivate || true
conda run -n isp python3 ${ISP_DIR}/setup_user.py build_ext --inplace

popd > /dev/null

# Create an alias for ISP in multiple shell configurations
read -p "Create alias for ISP in your shell configuration? [Y/n] " ALIAS_CHOICE
ALIAS_CHOICE=$(echo "$ALIAS_CHOICE" | tr '[:upper:]' '[:lower:]')
echo

if [[ "$ALIAS_CHOICE" == "y" || -z "$ALIAS_CHOICE" ]]; then
    CONFIG_FILES=(~/.bashrc ~/.bash_profile ~/.zshrc ~/.zprofile ~/.profile ~/.kshrc ~/.tcshrc ~/.cshrc ~/.config/fish/config.fish)

    for PROFILE in "${CONFIG_FILES[@]}"; do
        if [[ -f "$PROFILE" ]]; then
            if [[ "$(uname -s)" == "Darwin" ]]; then
                sed -i '' '/# ISP Installation/,/# ISP Installation end/d' "$PROFILE"
            else
                sed -i '/# ISP Installation/,/# ISP Installation end/d' "$PROFILE"
            fi
            echo "# ISP Installation" >> "$PROFILE"
            if [[ "$PROFILE" == *fish* ]]; then
                echo "alias isp '${ISP_DIR}/isp.sh'" >> "$PROFILE"
            elif [[ "$PROFILE" == *tcshrc* || "$PROFILE" == *cshrc* ]]; then
                echo "alias isp '${ISP_DIR}/isp.sh'" >> "$PROFILE"
            else
                echo "alias isp='${ISP_DIR}/isp.sh'" >> "$PROFILE"
            fi
            echo "# ISP Installation end" >> "$PROFILE"
            echo "Updated alias in $PROFILE"
        fi
    done

    echo "Aliases have been updated in your shell configuration files."
    echo "Run ISP by typing 'isp' in the terminal."
else
    echo "To run ISP, execute isp.sh at ${ISP_DIR}"
fi

# ================================
# ADDING DESKTOP SHORTCUT OPTION
# ================================
OS_TYPE=$(uname -s)

# macOS shortcut creation
if [[ $OS_TYPE == "Darwin" ]]; then
    echo "macOS Desktop shortcut creation"

    read -p "Do you want to create a desktop shortcut for ISP? [Y/n] " SHORTCUT
    SHORTCUT=$(echo "$SHORTCUT" | tr '[:upper:]' '[:lower:]')
    if [[ "$SHORTCUT" == "y" || -z "$SHORTCUT" ]]; then
        SHORTCUT_PATH=~/Desktop/ISP_Launcher.command
        
        echo "Creating desktop shortcut at $SHORTCUT_PATH..."
        cat <<EOF > "$SHORTCUT_PATH"

${ISP_DIR}/isp.sh

EOF

        chmod +x "$SHORTCUT_PATH"
        echo "Shortcut created! You can now double-click 'ISP_Launcher.command' on your Desktop to launch ISP."

        read -p "Would you like to create a macOS application shortcut instead? [Y/n] " APP_SHORTCUT
        APP_SHORTCUT=$(echo "$APP_SHORTCUT" | tr '[:upper:]' '[:lower:]')
        if [[ "$APP_SHORTCUT" == "y" || -z "$APP_SHORTCUT" ]]; then
            AUTOMATOR_SCRIPT_PATH=~/Desktop/ISP_Launcher.app
            osascript <<EOF
tell application "Automator"
    activate
    set myWorkflow to make new workflow
    tell myWorkflow
        set myAction to make new action at end with properties {name:"Run Shell Script"}
        set shell script of myAction to "bash '$SHORTCUT_PATH'"
    end tell
    save myWorkflow as application "$AUTOMATOR_SCRIPT_PATH"
end tell
EOF
            echo "Automator application shortcut created at $AUTOMATOR_SCRIPT_PATH!"
        fi
    fi
fi

echo "Installation complete!"

# Detect shell type
SHELL_NAME=$(basename "$SHELL")

# Choose appropriate rc file
if [ "$SHELL_NAME" = "zsh" ]; then
    RC_FILE=".zshrc"
elif [ "$SHELL_NAME" = "bash" ]; then
    RC_FILE=".bashrc"
elif [ "$SHELL_NAME" = "fish" ]; then
    RC_FILE=".config/fish/config.fish"
else
    RC_FILE="your shell configuration file"
fi

echo "Please run: source ~/$RC_FILE"
echo "Alternatively, close this terminal and reopen it, then type 'isp' to start."