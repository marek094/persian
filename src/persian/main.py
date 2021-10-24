import argparse
import sys
from typing import List, Iterator
from itertools import product
from numpy import arange


def dynamic_import(module, cls):
    imp = __import__(module, fromlist=[cls])
    return getattr(imp, cls)


def string_to_arggrid(astr: str) -> Iterator[List[str]]:
    """
    Each line has one argument, 
    multiple values are separated by semicolon
    """
    argset = []
    for x in astr.strip().split('\n'):
        k, *v = x.strip().split(' ', maxsplit=1)
        if len(v) == 1:
            for val in v[0].strip().split(';'):
                argset.append([k, val.strip()])
        else:
            argset.append([k])
    return product(*argset)


def string_to_args(astr: str) -> List[str]:
    """
    Each line has one argument
    """
    grid = list(string_to_arggrid(astr=astr))
    assert len(grid) == 1, "Arg value contains unexpected `;`"
    return grid[0]


def experiment_builder(parser_experiments=None,
                       verbose=True,
                       default_args=[],
                       args=None):
    parser_main = argparse.ArgumentParser()
    parser_main.add_argument('--schema',
                             type=str,
                             required=True,
                             help='for example `torch_dataset`')
    parser_main.add_argument('--print_help', action='store_true')

    args_all = default_args
    args_all += args if args is not None else sys.argv[1:]
    args_main, args_unknown = parser_main.parse_known_args(args=args_all)

    parts = args_main.schema.split('_')[::-1]
    schema_cls = "".join([x[:1].upper() + x[1:] for x in parts]) + 'Schema'
    DynSchema = dynamic_import(f'persian.schemas.{args_main.schema}',
                               schema_cls)

    # print('From', args_main.schema, 'importing', schema_cls)

    parser_schema = DynSchema.build_parser()
    if args_main.print_help:
        parser_main.print_help()
        parser_schema.print_help()
        if parser_experiments is not None:
            parser_experiments.print_help()

    if parser_experiments is None:
        args = parser_schema.parse_args(args=args_unknown)
        args_snd = None
    else:
        args, args_unknown_snd = parser_schema.parse_known_args(
            args=args_unknown)
        args_snd = parser_experiments.parser_args(args=args_unknown_snd)

    model = DynSchema.from_args(args)
    if verbose:
        print('Run:', model.as_hstr())
    return model, args_snd
