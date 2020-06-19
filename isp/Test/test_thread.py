import time

from isp.Utils import AsycTime


def print_return_background(result):
    print("Result from background: ", result)


@AsycTime.run_async()
def do_something(secs, msg=""):
    for i in range(2):
        time.sleep(secs)
        print("Task in background thread {}".format(i))
    print(msg)
    print('Done background task')


@AsycTime.run_async(print_return_background)
def do_something_with_return(secs):
    for i in range(2):
        time.sleep(secs)
        print("Task in background 2 thread {}".format(i))
    print('Done background 2 task')
    return 0


def do_something_else():
    for i in range(5):
        time.sleep(1)
        print("Task in main thread {}".format(i))


print("Program started")
do_something(2, "Hi")
do_something_with_return(2)
do_something_else()

# wait to finish script
time.sleep(5)
