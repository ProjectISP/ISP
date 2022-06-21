import sys
from isp.Gui import start_isp, except_hook

if __name__ == '__main__':
    print("WELCOME TO INTEGRATED SEISMIC PROGRAM 2022.6.1") 
    sys.excepthook = except_hook
    start_isp()
