import sys
from isp.Gui import start_isp, except_hook

if __name__ == '__main__':
    sys.excepthook = except_hook
    start_isp()
