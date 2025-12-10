import pymodbus
import sys
print("Current pymodbus version:", pymodbus.__version__)
print("Location:", pymodbus.__file__)
print("Python Executable:", sys.executable)
from .onrobot import RG

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

def open_gripper():
    gripper.open_gripper()

def close_gripper():
    gripper.close_gripper()

def main():
    close_gripper()
    open_gripper()
