import sys,subprocess

def install_packages():
  packages = [
    "beeai-framework",
    "requests",
    "beautifulsoup4",
    "numpy",
    "pandas",
    "pydantic"
  ]

  print("Installing required packages...")
  for package in packages:
    try:
      subprocess.check_call([sys.executable, "-m", "pip", "install", package])
      print(f"{package} installed succesfully")
    except subprocess.CalledProcessError as e:
      print(f"Failed to install {package}: {e}")

  print("Installation Completed")

try:
  install_packages()
  from beeai_framework.backend.chat import ChatModel
  from beeai_framework.agents import Agent
  from beeai_framework.tools import Tool
  from beeai_framework.workflows import Workflow

  BEEAI_AVAILABLE = True
  print(f"BeeAI Framework imported successfully")
except ImportError as e:
  print(f"BeeAI Framework import failed: {e}")
  print(f"Failling back to custom implementation...")
  BEEAI_AVAILABLE = False