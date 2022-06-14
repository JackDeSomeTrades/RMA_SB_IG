
# Taken from https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python/66815839#66815839
class DynamicClassCreator():
    def __init__(self):
        self.created_classes = {}

    def __call__(self, *bases):
        rep = ",".join([i.__name__ for i in bases])
        if rep in self.created_classes:
            return self.created_classes[rep]

        class MyCode(*bases):
            pass
        self.created_classes[rep] = MyCode

        return MyCode


ForwardClassFactory = DynamicClassCreator()