class Parameter:
    def __init__(self) -> None:
        self.data = {}

    def set(self, names_or_parameter, values=None):
        if isinstance(names_or_parameter, Parameter):
            self.data = names_or_parameter.data
        elif values is not None:
            for key, value in zip(names_or_parameter, values):
                self.data[key] = value
        else:
            raise ValueError("Invalid arguments")

    def get(self, names):
        # note this can be a list of names
        return [self.data[name] for name in names]

    def keys(self):
        return self.data.keys()
