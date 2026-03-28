import subprocess

from enum import Enum
import platform


class PYTHON_INSTALLED(Enum):
    PYTHON = 1
    PYTHON_3 = 2
    NOT_INSTALLED = 3


def python_version():
    result = is_python_installed()
    if result is PYTHON_INSTALLED.PYTHON_3:
        return "python3"
    elif result is PYTHON_INSTALLED.PYTHON:
        return "python"
    else:
        return ""


def is_python_installed():
    python_installed = check_python_version("python")
    python3_installed = check_python_version("python3")
    if python3_installed:
        return PYTHON_INSTALLED.PYTHON_3
    elif python_installed:
        return PYTHON_INSTALLED.PYTHON
    else:
        return PYTHON_INSTALLED.NOT_INSTALLED


def check_python_version(version):
    try:
        subprocess.run([version, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def is_wsl():
    """
        Check if using Windows Subsystem for Linux
    """
    if platform.system() != 'Linux':
        return False

    try:
        with open('/proc/version', 'r') as f:
            content = f.read().lower()
            return 'microsoft' in content or 'wsl' in content
    except FileNotFoundError:
        return False
