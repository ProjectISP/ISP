import functools
from multiprocessing.dummy import Pool as ThreadPool

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, pyqtBoundSignal


class Worker(QObject):
    finished: pyqtBoundSignal = pyqtSignal(object)
    progress: pyqtBoundSignal = pyqtSignal(int)
    exception: pyqtBoundSignal = pyqtSignal(object)


    def __init__(self):
        super().__init__()
        self.__func = None

        self.thread = QThread()
        self.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.__run)
        self.finished.connect(self.thread.quit)
        self.finished.connect(self.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

    def start(self):
        self.thread.start()

    def job_finished(self, func):
        self.finished.connect(func)

    def job(self, func):
        self.__func = func

    def emmit_progress(self, v):
        self.progress.emit(v)

    def progress_callback(self, func):
        self.progress.connect(func)

    def register_exception(self, func):
        self.exception.connect(func)

    @pyqtSlot()
    def __run(self):
        """Long-running task."""
        result = None
        try:
            if self.__func:
                result = self.__func()
        except Exception as e:
            print(f"Exception at worker: {self}")
            print(str(e))
            self.exception.emit(e)
        finally:
            self.finished.emit(result)
            self.thread.exit()


class ParallelWorkers(QObject):
    finished: pyqtBoundSignal = pyqtSignal()
    progress: pyqtBoundSignal = pyqtSignal(int, object)
    exception: pyqtBoundSignal = pyqtSignal(object)

    def __init__(self, threads: int):
        super().__init__()
        self.threads = threads
        self.__func = None
        self.__inter = None
        self.__k = {}

        self.thread = QThread()
        self.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.__run)
        self.finished.connect(self.thread.quit)
        self.finished.connect(self.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

    def start(self, inter, **kwargs):
        self.__inter = inter
        self.__k = kwargs
        self.thread.start()

    def job_finished(self, func):
        self.finished.connect(func)

    def job(self, func):
        self.__func = func

    def progress_callback(self, func):
        self.progress.connect(func)

    def register_exception(self, func):
        self.exception.connect(func)

    @pyqtSlot()
    def __run(self):
        try:
            inter = self.__inter
            kwargs = self.__k
            with ThreadPool(self.threads) as pool:
                r = [pool.apply_async(self.__func, args=(v, ), kwds=kwargs) for v in inter]
                total = len(r)
                for i, p in enumerate(r):
                    pg = (i + 1) / total * 100
                    result = p.get()
                    self.progress.emit(pg, result)

        except Exception as e:
            print(f"Exception at ParallelWorkers: {self}")
            print(str(e))
            self.exception.emit(e)
        finally:
            self.finished.emit()
            self.thread.exit()

def thread(on_finished: str = None):
    def decorator_thread(func):
        @functools.wraps(func)
        def wrapper_thread(*args, **kwargs):
            self = args[0]  # expected to be an object
            on_finished_func = self.__getattribute__(on_finished)
            worker_name = f"worker_{func.__name__}"
            self.__setattr__(worker_name, Worker())  # sets a new worker for this method
            worker = self.__getattribute__(worker_name)

            worker.job(lambda: func(*args, **kwargs))
            worker.job_finished(on_finished_func)
            worker.start()

        return wrapper_thread
    return decorator_thread