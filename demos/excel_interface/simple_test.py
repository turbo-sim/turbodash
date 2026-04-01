import xlwings as xw


print("Running simple_test.py")

@xw.func
def py_sum(a, b):
    print("py_sum called with:", a, b)
    return a + b

