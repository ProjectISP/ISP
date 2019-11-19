import time
from functools import wraps
from threading import Thread
from timeit import Timer


class AsycTime(Thread):

    def __init__(self):
        super().__init__()
        self.seconds = 1
        self.__callback = None

    def run(self) -> None:
        time.sleep(self.seconds)
        if self.__callback:
            self.__callback()

    def wait(self, seconds: float, func):
        """
        Wait x seconds in a different thread.

        :param seconds: Time in seconds to wait.
        :param func: The callback function. This will be called when the waiting time is over.
        :return:
        """
        self.seconds = seconds
        self.start()
        self.__callback = func

    @staticmethod
    def async_wait(seconds: float):
        """
         Wait x seconds in a different thread before execute the decorated method.

        :param seconds: seconds: Time in seconds to wait.
        :return:
        """
        def app_decorator(func):
            @wraps(func)
            def wrap_func(*args, **kwargs):
                AsycTime().wait(seconds, lambda: func(*args, **kwargs))
            return wrap_func
        return app_decorator


def time_method(loop=10000):
    def timer_decorator(func):
        @wraps(func)
        def wrap_func(*args, **kwargs):
            total_time = Timer(lambda: func(*args, **kwargs)).timeit(number=loop)
            print("Method {name} run {loop} times".format(name=func.__name__, loop=loop))
            print("It took: {time} s, Mean: {mean_time} s".
                  format(mean_time=total_time / loop, time=total_time))
            return func
        return wrap_func
    return timer_decorator
