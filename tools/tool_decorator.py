def tool(func):
    func.name = func.__name__
    return func
