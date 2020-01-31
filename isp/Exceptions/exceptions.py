from functools import wraps
from subprocess import CalledProcessError, SubprocessError


def parse_excepts(msg_callback):
    """
    Wrapper for catching excepts. If one of the flowing excepts is caught the error message will
    be passed to the msg_callback method.

    This method will catch the following excepts: FileNotFoundError, AttributeError, CalledProcessError, SubprocessError

    Example::

        @parse_excepts(lambda self, msg: self.func(msg))
        def run_subprocess(self):

    :param msg_callback: A callback to print the exception messages. Method must have a msg parameter.
    :return:
    """
    def wrapper(func):
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            msg = None
            try:
                func(self, *args, **kwargs)
            except (FileNotFoundError, AttributeError) as e:
                msg = "{}".format(e)
            except CalledProcessError as e:
                # bad error
                msg = "Error code {} when trying to run subprocess. {}".format(e.returncode, e)
            except SubprocessError as e:
                msg = "{}".format(e)
            finally:
                msg_callback(self, msg)

        return wrapped
    return wrapper


# Exceptions Classes
class InvalidFile(Exception):

    def __init__(self, message):
        self.message = message
        Exception.__init__(self, message)
