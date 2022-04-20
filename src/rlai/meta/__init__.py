import importlib
import inspect
import os
import pkgutil
from typing import Set, Dict, List, Union, Callable, Any

import requests

import rlai


def rl_text(
        chapter: Union[int, str],
        page: int
) -> Callable:
    """
    Decorator for RL text references.

    :param chapter: Either an integer chapter within RL text that describes the Python element being decorated, or a
    string chapter about something else.
    :param page: Page within RL text that describes the Python element being decorated.
    :return: Decorator function.
    """

    def inner(
            element: Any
    ) -> Any:

        if os.getenv('ANNOTATIONS_ON', 'False') != 'True':
            return element

        element.rl_text_chapter = chapter
        element.rl_text_page = page
        element.rl_text_description = f'RL 2nd Edition, ch. {chapter}, p. {page}'

        return element

    return inner


def summarize(
        pkg: Any,
        chapter_page_descriptions: Dict[int, Dict[int, List[str]]],
        paths_summarized: Set[str] = None,
):
    """
    Summarize all code decorated.

    :param pkg: Top-level package.
    :param chapter_page_descriptions: Chapter page descriptions.
    :param paths_summarized: Paths summarized so far.
    """

    if paths_summarized is None:
        paths_summarized = set()

    repo_src_url = 'https://github.com/MatthewGerber/rlai/tree/master/src/'

    for _, module_path, module_is_pkg in pkgutil.iter_modules(pkg.__path__, prefix=pkg.__name__ + '.'):

        module = importlib.import_module(module_path)

        for attribute_name in dir(module):
            if not attribute_name.startswith('_') and hasattr(module, attribute_name):
                attribute = getattr(module, attribute_name)
                if hasattr(attribute, 'rl_text_description'):
                    full_path = f'{attribute.__module__}.{attribute.__name__}'
                    if full_path not in paths_summarized:

                        chapter = attribute.rl_text_chapter
                        page = attribute.rl_text_page

                        if chapter not in chapter_page_descriptions:
                            chapter_page_descriptions[chapter] = {}

                        if page not in chapter_page_descriptions[chapter]:
                            chapter_page_descriptions[chapter][page] = []

                        # inspect is unpredictable, in that it sometimes returns the correct line and sometimes returns
                        # the line previous to the line with the attribute. check each.
                        src_lines, line_no = inspect.findsource(attribute)
                        if attribute.__name__ not in src_lines[line_no]:
                            if attribute.__name__ in src_lines[line_no + 1]:
                                line_no = line_no + 1
                            else:
                                raise ValueError('Attribute name not found on src lines.')

                        # inspect returns 0-based line number of decorator. move to next 1-based line for url.
                        line_no += 1

                        relative_url_filepath = full_path.replace(".", "/").rsplit("/", maxsplit=1)[0]
                        if module.__file__.endswith('__init__.py'):
                            relative_url_filepath = f'{relative_url_filepath}/__init__'

                        src_url = f'{repo_src_url}{relative_url_filepath}.py#L{line_no}'
                        response = requests.get(src_url)
                        if response.status_code == 200:
                            chapter_page_descriptions[chapter][page].append(f'### [{full_path}]({src_url})\n```\n{attribute.__doc__.strip()}\n```\n')
                            paths_summarized.add(full_path)
                        else:
                            print(f'Invalid URL:  {src_url}')

        if module_is_pkg:
            summarize(module, chapter_page_descriptions, paths_summarized)


def main():

    chapter_page_descriptions = {}

    # noinspection PyTypeChecker
    summarize(rlai, chapter_page_descriptions)

    docs_dir = f'{os.path.dirname(__file__)}/../../../docs/'
    meta_md_path = f'{docs_dir}index.md'

    ch_num_name = {
        1: 'Introduction',
        2: 'Multi-armed Bandits',
        3: 'Finite Markov Decision Processes',
        4: 'Dynamic Programming',
        5: 'Monte Carlo Methods',
        6: 'Temporal-Difference Learning',
        7: 'n-step Bootstrapping',
        8: 'Planning and Learning with Tabular Methods',
        9: 'On-policy Prediction with Approximation',
        10: 'On-policy Control with Approximation',
        11: 'Off-policy Methods with Approximation',
        12: 'Eligibility Traces',
        13: 'Policy Gradient Methods',
        14: 'Psychology',
        15: 'Neuroscience',
        16: 'Applications and Case Studies',
        17: 'Frontiers'
    }

    # read base index
    with open('index_base.md', 'r') as base:
        readme_base = base.read().strip()

    # write markdown file
    with open(meta_md_path, 'w') as meta_md:

        meta_md.write(f'{readme_base}\n\n')

        # write sorted-string chapters
        meta_md.write('# Links to Code by Topic\n')
        for chapter in sorted(filter(lambda ch: isinstance(ch, str), chapter_page_descriptions)):
            ch_filename = f'ch_{chapter.replace(" ", "_")}.md'
            meta_md.write(f'### [{chapter}]({ch_filename})\n')
            with open(f'{docs_dir}{ch_filename}', 'w') as ch_md:
                ch_md.write(f'[Home](index.md)\n\n# {chapter}\n')
                for page in sorted(chapter_page_descriptions[chapter]):
                    for description in sorted(chapter_page_descriptions[chapter][page]):
                        ch_md.write(description)

        # write sorted numeric chapters
        meta_md.write('\n# Links to Code by Book Chapter\n')
        for chapter in sorted(filter(lambda ch: isinstance(ch, int), chapter_page_descriptions)):
            ch_filename = f'ch_{chapter}.md'
            meta_md.write(f'### [Chapter {chapter}:  {ch_num_name[chapter]}]({ch_filename})\n')
            with open(f'{docs_dir}{ch_filename}', 'w') as ch_md:
                ch_md.write(f'[Home](index.md)\n\n# Chapter {chapter}:  {ch_num_name[chapter]}\n')
                for page in sorted(chapter_page_descriptions[chapter]):
                    for description in sorted(chapter_page_descriptions[chapter][page]):
                        ch_md.write(description)


if __name__ == '__main__':
    main()
