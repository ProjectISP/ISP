import time

from isp.Utils import AsycTime

class DoStuff:

    def __init__(self):
        pass

    def print_return_background(self, result):
        print("Result from background: ", result)


    @AsycTime.run_async()
    def do_something(self, secs, msg=""):
        for i in range(2):
            time.sleep(secs)
            print("Task in background thread {}".format(i))
        print(msg)
        print('Done background task')


    @AsycTime.run_async(lambda self, r: self.print_return_background(r))
    def do_something_with_return(self, secs):
        for i in range(2):
            time.sleep(secs)
            print("Task in background 2 thread {}".format(i))
        print('Done background 2 task')
        return 0


    def do_something_else(self):
        for i in range(5):
            time.sleep(1)
            print("Task in main thread {}".format(i))


print("Program started")
stuff = DoStuff()
stuff.do_something(2, "Hi")
stuff.do_something_with_return(2)
stuff.do_something_else()

# wait to finish script
time.sleep(5)
