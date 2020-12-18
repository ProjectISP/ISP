

class Singleton:
    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwargs):
        if not self.instance:
            self.instance = self.klass(*args, **kwargs)
        return self.instance
