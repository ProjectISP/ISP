import sys
import time

from PyQt5.QtCore import QThreadPool

from isp.Gui import pw
from isp.Gui.Utils import Worker


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

    def __init__(self):
        super().__init__()

        # set simple UI
        button_start_worker = pw.QPushButton(self)
        button_start_worker.setText("Start worker")
        button_start_worker.clicked.connect(self.start_job)
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


if __name__ == '__main__':
    start_app()
