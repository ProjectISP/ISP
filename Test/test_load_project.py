import pickle


def read_load_project(file):
    project = pickle.load(open(file, "rb"))
    return project

file ='/Users/robertocabieces/Documents/desarrollo/ISP2021/isp/examples/Moment_Tensor_example/MTI'
project = read_load_project(file)
print(project)