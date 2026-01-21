class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self, module_class):
        self._module_dict[module_class.__name__] = module_class
        return module_class

    def get(self, key):
        return self._module_dict.get(key)

# Khởi tạo các "sổ cái" quản lý
MODELS = Registry('models')