import subprocess

from enum import Enum
import platform


class PYTHON_INSTALLED(Enum):
    PYTHON = 1
    PYTHON_3 = 2
    NOT_INSTALLED = 2


def python_version():
    python_installed = is_python_installed()
    if python_installed.PYTHON_3:
        return "python3"
    elif python_installed.PYTHON:
        return "python"
    else:
        return ""


def is_python_installed():
    # Check for Python (may be Python 2)
    python_installed = check_python_version("python")
    # Check for Python 3
    python3_installed = check_python_version("python3")
    if python3_installed:
        return PYTHON_INSTALLED.PYTHON_3
    elif python_installed:
        return PYTHON_INSTALLED.PYTHON
    else:
        return PYTHON_INSTALLED.NOT_INSTALLED


def check_python_version(version):
    try:
        # Run the command to check for Python version
        subprocess.run([version, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(f"{version} is installed.")
        return True
    except subprocess.CalledProcessError:
        # print(f"{version} is not installed.")
        return False
    except FileNotFoundError:
        # print(f"{version} is not installed.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def is_wsl():
    """
    Check if the current system is Windows Subsystem for Linux (WSL).
    :return:
    """
    # Check the platform identifier
    if platform.system() != 'Linux':
        return False

    # Read the contents of /proc/version
    try:
        with open('/proc/version', 'r') as f:
            content = f.read().lower()
            # Look for keywords indicating WSL
            return 'microsoft' in content or 'wsl' in content
    except FileNotFoundError:
        return False
