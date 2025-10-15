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
conda deactivate
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
echo  "Please  >> source .bashrc or source .zshrc, then type isp in your terminal"
echo  "Alternatively close terminal and open it, then type isp in your terminal"