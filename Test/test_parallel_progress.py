from concurrent.futures.thread import ThreadPoolExecutor
import PyQt5.QtCore as pyc
import PyQt5.QtWidgets as pw

from time import sleep
import sys


class Test(pyc.QObject):
    # Sign must be defined inside of a QObject
    sig_set_value = pyc.pyqtSignal(int)
    sig_exit = pyc.pyqtSignal()
    cancelled = True

    def cancelled_callback(self):
        self.cancelled = True

    def defined_progress_callback(self):
        for i in range(50):
            if self.cancelled:
                return
            sleep(1)
            self.sig_set_value.emit(i)
        self.sig_set_value.emit(50)

    def defined_progress(self):
        self.cancelled = False
        parallel_progress_run("Current progress: ", 0, 50, None,
                              self.defined_progress_callback, self.cancelled_callback,
                              signalValue=self.sig_set_value)

    def undefined_progress_callback(self):
        for i in range(50):
            if self.cancelled:
                return
            sleep(1)
        self.sig_exit.emit()

    def undefined_progress(self):
        self.cancelled = False
        parallel_progress_run("Current progress:", 0,0, None, self.undefined_progress_callback,
                              self.cancelled_callback, signalExit=self.sig_exit)


def parallel_progress_run(text, min_val, max_val, parent, callback, cancel_callback=None, **kwargs):
    pgbar = pw.QProgressDialog(text, "Cancel", min_val, max_val, parent)
    pgbar.setValue(min_val)
    if (min_val == 0 and max_val == 0 and 'signalExit' in kwargs
            and hasattr(kwargs['signalExit'], 'connect')):
        kwargs['signalExit'].connect(pgbar.accept)
    elif ('signalValue' in kwargs and
          hasattr(kwargs['signalValue'], 'connect')):
        kwargs['signalValue'].connect(pgbar.setValue)
    else:
        raise Exception
    with ThreadPoolExecutor(1) as executor:
        f = executor.submit(callback)
        pgbar.exec()
        f.cancel()
        if cancel_callback is not None:
            cancel_callback()


if __name__ == "__main__":
    app = pw.QApplication(sys.argv)
    t = Test()
    t.defined_progress()
    t.undefined_progress()
