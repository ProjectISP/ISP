import sys
import time
from typing import List

from PyQt5.QtCore import QThreadPool, pyqtSlot, pyqtBoundSignal, pyqtSignal

from isp.Gui import pw
from isp.Gui.Utils import Worker
from isp.Gui.Utils.pyqt_threading_utils import thread, ParallelWorkers


class Controller:

    def __init__(self):

        self.main_window = None

    def open_main_window(self):
        self.main_window = MainWindow()
        self.main_window.show()


def start_app():
    app = pw.QApplication(sys.argv)
    controller = Controller()
    controller.open_main_window()
    sys.exit(app.exec_())


class MainWindow(pw.QMainWindow):

    THREADS = QThreadPool.globalInstance().maxThreadCount()
    plot_data_signal: pyqtBoundSignal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.workers = None
        # set simple UI
        button_start_worker = pw.QPushButton(self)
        button_start_worker.setText("Start worker")
        self.plot_data_signal.connect(self.plot_data)
        #button_start_worker.clicked.connect(self.start_job)
        button_start_worker.clicked.connect(self.on_click_button)
        button_start_worker.move(50, 20)

        self.worker = None

    def start_job(self):
        self.worker = Worker()
        # In this example long_job has one enter parameter, if it didn't need any parameter the call
        # could be simplify by: self.worker.job(self.long_job)
        self.worker.job(lambda: self.long_job('hard worker'))
        # In this example long_job return a value so job_finished must have its return value register
        # If long_job doesn't return anything this can be simplify by: self.worker.job_finished(self.job_finished)
        self.worker.job_finished(lambda name: self.job_finished(name))

        # if you need to keep track of progress you can register a callback method like this. However,
        # you must emmit the progress yourself since this is A single thread worker.
        # For multi-thread check: ParallelWorkers class
        self.worker.progress_callback(lambda p: print(f'progress {p}')) # progress method could be hooked with a progress bar
        # starts the work in a different thread.
        self.worker.start()
        print('Continue on main thread')

    def long_job(self, job_name: str):
        print(f'{job_name} started')
        for i in range(5):
            # that's how you can emmit progress with a single thread worker.
            self.worker.emmit_progress(i)
            time.sleep(1)
        return job_name

    def job_finished(self, name: str):
        print(f'{name} finished')


    def on_click_button(self):
        self.workers = ParallelWorkers(4)
        # Here we can disabled thing or make additional staff
        self.workers.job(self.process_data)
        # All jobs finished
        self.workers.progress_callback(self.process_data_progress)
        self.workers.job_finished(self.process_data_finished)

        self.workers.start([f"file_{i}" for i in range(100)])

    #@thread("process_data_finished")
    def process_data(self, file_name: str):
        print(f"Process data started. {file_name}")
        time.sleep(1)
        data = [1, 2, 3]
        self.plot_data_signal.emit(data)
        return data

    @pyqtSlot(int, object)
    def process_data_progress(self, p: int, data: List[int]):
        # GUI STAFF MUST BE HERE
        print(data)

    @pyqtSlot() #for 1 thread
    def process_data_finished(self):
        print("work done")

    @pyqtSlot(object)
    def plot_data(self, data: List[int]):
        print("Plotting data")




if __name__ == '__main__':
    start_app()
