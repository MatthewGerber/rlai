import importlib
import pkgutil
from typing import Set


def rl_text(
        chapter: int,
        page: int
):
    """
    Decorator for RL text references.

    :param chapter: Chapter within RL text that describes the Python element being decorated.
    :param page: Page within RL text that describes the Python element being decorated.
    :return: Decorator function.
    """

    def inner(func):

        func.rl_text_description = f'RL 2nd Edition, ch. {chapter}, p. {page}'

        return func

    return inner


def summarize(
        pkg,
        paths_summarized: Set[str] = None
):
    """
    Summarize all code decorated.

    :param pkg: Top-level package.
    :param paths_summarized: Paths summarized so far.
    """

    if paths_summarized is None:
        paths_summarized = set()

    for _, module_path, module_is_pkg in pkgutil.iter_modules(pkg.__path__, prefix=pkg.__name__ + '.'):

        module = importlib.import_module(module_path)

        for attribute_name in dir(module):
            if not attribute_name.startswith('_') and hasattr(module, attribute_name):
                attribute = getattr(module, attribute_name)
                if hasattr(attribute, 'rl_text_description'):
                    full_path = f'{attribute.__module__}.{attribute.__name__}'
                    if full_path not in paths_summarized:
                        print(f'{full_path} ({attribute.rl_text_description}):{attribute.__doc__}')
                        paths_summarized.add(full_path)

        if module_is_pkg:
            summarize(module, paths_summarized)
