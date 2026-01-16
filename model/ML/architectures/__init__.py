import importlib
import dataclasses
import typing
import sys


@dataclasses.dataclass
class ModuleFactory:
    module: str
    class_name: str
    fixed_args: None | dict[str, typing.Any] = None
    reload: bool = True

    def __call__(self, *args, **kwargs):
        fixed_args = {} if self.fixed_args is None else self.fixed_args

        full_module_name = f"{__name__}.{self.module}"

        if full_module_name in sys.modules:
            module = sys.modules[full_module_name]
            if self.reload:
                module = importlib.reload(module)
        else:
            module = importlib.import_module(f".{self.module}", __name__) #hihi
        return getattr(module, self.class_name)(*args, **kwargs, **fixed_args)

    def __repr__(self):
        fixed_str = f", fixed_args={self.fixed_args!r}" if self.fixed_args else ""
        return f"ModuleFactory({__name__}.{self.module}:{self.class_name}{fixed_str})"



ARCHITECTURES = {
    "cnn": ModuleFactory("cnn", "CNN"),
    "zero": ModuleFactory("zero", "ZeroModel"),
}


def net_constructor(arch):
    if (constructor := ARCHITECTURES.get(arch)) is not None:
        return constructor
    raise ValueError(f"unknown architecture {arch}")