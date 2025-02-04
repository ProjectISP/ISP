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

# Function to detect OS and create Conda environment
create_isp_env() {
  OS_TYPE=$(uname -s)
  echo "Operating System detected: $OS_TYPE"

  if [[ $OS_TYPE == "Darwin" ]]; then
    echo "MacOS detected. Checking processor type..."
    
    # Determine processor type
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

#echo "Updating Conda to the latest version..."
#conda update -n base -c defaults conda -y

# Check if the ISP environment exists
if conda env list | grep -q '^isp\s'; then
  echo "'isp' environment already exists."
  read -p "Do you want to remove the existing environment and reinstall ISP? (Y/N): " CHOICE
  case "$CHOICE" in
    [Yy]* )
      echo "Removing existing 'isp' environment..."
      conda remove --name isp --all -y
      echo "Reinstalling ISP environment..."
      create_isp_env
      ;;
    * ) 
      echo "Skipping environment reinstallation. Proceeding with the existing setup."
      ;;
  esac
else
  echo "No 'isp' environment found. Proceeding to create one."
  create_isp_env
fi

echo "ISP environment process finished"

# Compile packages
pushd ${ISP_DIR} > /dev/null

# Select installation type
read -p 'Which type of installation would you prefer, conventional or advanced ? ' REPLY

if [[ $REPLY = 'conventional' ]]
then
    source activate isp
    python3 ${ISP_DIR}/setup_user.py build_ext --inplace
elif [[ $REPLY = 'advanced' ]]
then
    source activate isp
    python3 ${ISP_DIR}/setup_developer.py build_ext --inplace
fi
popd > /dev/null

# Create an alias for ISP in multiple shell configurations
read -p "Create alias for ISP in your shell configuration? [Y/n] " ALIAS_CHOICE
echo    # Move to a new line

if [[ $ALIAS_CHOICE =~ ^[Yy]$ ]]; then
    CONFIG_FILES=(~/.bashrc ~/.bash_profile ~/.zshrc ~/.zprofile ~/.profile ~/.kshrc ~/.tcshrc ~/.cshrc ~/.config/fish/config.fish)

    for PROFILE in "${CONFIG_FILES[@]}"; do
        if [[ -f "$PROFILE" ]]; then
            echo "# ISP Installation" >> "$PROFILE"
            if [[ "$PROFILE" == *fish* ]]; then
                echo "alias isp '${ISP_DIR}/isp.sh'" >> "$PROFILE"
            elif [[ "$PROFILE" == *tcshrc* || "$PROFILE" == *cshrc* ]]; then
                echo "alias isp '${ISP_DIR}/isp.sh'" >> "$PROFILE"
            else
                echo "alias isp=${ISP_DIR}/isp.sh" >> "$PROFILE"
            fi
            echo "# ISP Installation end" >> "$PROFILE"
        fi
    done

    echo "Aliases have been added to your shell configuration files."
    echo "Run ISP by typing 'isp' in the terminal."
else
    echo "To run ISP, execute isp.sh at ${ISP_DIR}"
fi

#!/bin/bash

# ================================
# ADDING DESKTOP SHORTCUT OPTION
# ================================
OS_TYPE=$(uname -s)

# macOS shortcut creation
if [[ $OS_TYPE == "Darwin" ]]; then
    echo "macOS Desktop shortcut creation"

    read -p "Do you want to create a desktop shortcut for ISP? [Y/n] " SHORTCUT
    if [[ $SHORTCUT =~ ^[Yy]$ ]]; then
        SHORTCUT_PATH=~/Desktop/ISP_Launcher.command
        
        # Create the .command file
        echo "Creating desktop shortcut at $SHORTCUT_PATH..."
        cat <<EOF > "$SHORTCUT_PATH"

source ~/opt/anaconda3/etc/profile.d/conda.sh

source activate isp

${ISP_DIR}/isp.sh

EOF

        # Make it executable
        chmod +x "$SHORTCUT_PATH"
        echo "Shortcut created! You can now double-click 'ISP_Launcher.command' on your Desktop to launch ISP."

        # Optional: Make it a macOS application using Automator
        read -p "Would you like to create a macOS application shortcut instead? [Y/n] " APP_SHORTCUT
        if [[ $APP_SHORTCUT =~ ^[Yy]$ ]]; then
            AUTOMATOR_SCRIPT_PATH=~/Desktop/ISP_Launcher.app

            # Create AppleScript for Automator to make the app
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

# Linux shortcut creation
elif [[ $OS_TYPE == "Linux" ]]; then
    echo "Linux Desktop shortcut creation"

    read -p "Do you want to create a desktop shortcut for ISP? [Y/n] " SHORTCUT
    if [[ $SHORTCUT =~ ^[Yy]$ ]]; then
        SHORTCUT_PATH=~/Desktop/ISP_Launcher.desktop
        
        # Create the .desktop file
        echo "Creating desktop shortcut at $SHORTCUT_PATH..."
        cat <<EOF > "$SHORTCUT_PATH"
[Desktop Entry]
Version=2.0
Name=ISP
Comment=Launch ISP Application
Exec=bash -c "source ~/opt/anaconda3/etc/profile.d/conda.sh && conda activate isp && ${ISP_DIR}/isp.sh"
#Icon=${ISP_DIR}/isp/resources/images/LOGO.png  # Adjust this path to your icon
Terminal=true
Type=Application
Categories=Utility;Application;
EOF

        # Make it executable
        chmod +x "$SHORTCUT_PATH"
        echo "Shortcut created! You can now double-click 'ISP_Launcher.desktop' on your Desktop to launch ISP."
    fi

else
    echo "Unsupported operating system: $OS_TYPE"
fi

echo "Installation complete!"
