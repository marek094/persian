import sys
from typing import List
from argparse import ArgumentParser, BooleanOptionalAction
from numpy import arange

if sys.version_info.minor >= 9:
    from typing import Protocol
else:
    Protocol = object


class Schemable(Protocol):
    """
    """
    @staticmethod
    def list_hparams() -> List[dict]:
        """
            TODO: format specification
        """
        ...

    flags = None
    metrics = {}

    def __init__(self, flags={}) -> None:
        ...

    def prepare_dataset(self, set_name) -> None:
        ...

    def prepare_model(self) -> None:
        ...

    def prepare_criterium(self) -> None:
        ...

    def epoch_range(self) -> range:
        ...

    def run_batches(self, set_name) -> None:
        ...

    def run_inference(self, input):  # -> result
        ...

    def update_infoboard(self) -> None:
        ...

    def pack_model_params(self):  # -> packed_model
        ...

    def load_model_params(self) -> None:
        ...

    # directly implemented bellow in Schema class
    def metrics_report(self, set_name) -> str:
        ...

    def as_hstr(self) -> str:
        ...

    @classmethod
    def short_hparams(Cls) -> dict:
        ...

    @classmethod
    def build_parser(Cls) -> ArgumentParser:
        ...

    @classmethod
    def from_str(Cls, full_hstr) -> Protocol:
        ...

    @classmethod
    def from_args(Cls, args) -> Protocol:
        ...

    @classmethod
    def hgrid_gen(Cls) -> list:
        ...


class Schema(Schemable):
    @staticmethod
    def list_hparams() -> List[dict]:
        return []

    @classmethod
    def short_hparams(Cls):
        """
            Deterministic if hparams are added in the end
        """
        alphabet = "".join(chr(v) for v in range(ord('A'), ord('Z') + 1))
        alphabet += "".join(chr(v) for v in range(ord('a'), ord('z') + 1))

        shorts, used = {}, {'h': 0}

        def to_short(name, **_):
            for sh in name + alphabet:
                if sh not in used and sh in alphabet:
                    shorts[name] = sh
                    used[sh] = True
                    return

            shorts[name] = '0'

        for p in Cls.list_hparams():
            to_short(**p)
        return shorts

    def as_hstr(self):
        shorts = self.short_hparams()

        def to_elem(name, val):
            assert name in shorts
            short = shorts[name]
            return f"{short}{val}"

        hstr = ",".join(to_elem(*it) for it in self.flags.items())
        cls = type(self).__name__
        return f"{cls}_{hstr}"

    @classmethod
    def build_parser(Cls, **kwargs) -> ArgumentParser:
        shorts = Cls.short_hparams()
        parser = ArgumentParser(**kwargs)
        for param in Cls.list_hparams():
            param['help'] = ','.join([
                f'{d}: {param.pop(d)}' for d in ['range', 'help'] if d in param
            ])
            name = param.pop('name', "")
            names = [f'-{shorts[name]}', f'--{name}']
            if param['type'] == bool:
                if sys.version_info.minor >= 9:
                    param['action'] = BooleanOptionalAction
                else:
                    param2 = param.copy()
                    names2 = [f'--no-{name}']
                    param2['action'] = 'store_false'
                    param2['dest'] = name
                    parser.add_argument(*names2, **param2)

                    param['action'] = 'store_true'
            parser.add_argument(*names, **param)
        return parser

    @classmethod
    def from_args(Cls, args):
        argsd = args.__dict__
        flags = {}

        def to_flag(name, **_):
            if name in argsd:
                flags[name] = argsd[name]

        for p in Cls.list_hparams():
            to_flag(**p)
        return Cls(flags)

    @classmethod
    def hstr_as_flags(Cls, full_hstr):
        hparams = Cls.list_hparams()
        shorts = Cls.short_hparams()
        inv_shorts = {v: k for k, v in shorts.items()}
        ps = full_hstr.split('_')
        hstr = ps[1] if len(ps) > 1 else ps[0]
        flags = {}
        for item in hstr.split(','):
            k, val = item[:1], item[1:]
            name = inv_shorts[k]
            type_ = next(x['type'] for x in hparams if x['name'] == name)
            flags[name] = type_(val)
            # print(name, type_(val), type_, val, '#')
        return flags

    @classmethod
    def from_hstr(Cls, full_hstr):
        flags = Cls.hstr_as_flags(full_hstr)
        return Cls(flags)

    @classmethod
    def hgrid_gen(Cls):
        lazy = [(param['name'], arange(
            *param['range']) if 'range' in param else [param['default']])
                for param in reversed(Cls.list_hparams())]

        def gen(head, *tail):
            for flags in (gen(*tail) if len(tail) > 0 else [{}]):
                for value in head[1]:
                    yield {head[0]: value, **flags}

        return gen(*lazy)

    def metrics_report(self, set_name):
        return ", ".join(f'{k}: {v:f}'
                         for k, v in self.metrics[set_name].items())

    def __init__(self, flags={}):
        # create flags
        def to_flag(name, type, default, **_):
            if name in flags:
                val = flags[name]
                if val is None or val == 'None':
                    return name, None
                return name, type(val)
            return name, default

        hparams = self.list_hparams()
        self.flags = dict(to_flag(**p) for p in hparams)


SchemaSchema = Schema
