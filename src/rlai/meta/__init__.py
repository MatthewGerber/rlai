import importlib
import os
import pkgutil
from typing import Set, Dict, List

import md_toc

import rlai


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

    def inner(element):

        if os.getenv('ANNOTATIONS_ON', 'False') != 'True':
            return element

        element.rl_text_chapter = chapter
        element.rl_text_page = page
        element.rl_text_description = f'RL 2nd Edition, ch. {chapter}, p. {page}'

        return element

    return inner


def summarize(
        pkg,
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

    for _, module_path, module_is_pkg in pkgutil.iter_modules(pkg.__path__, prefix=pkg.__name__ + '.'):

        module = importlib.import_module(module_path)

        for attribute_name in dir(module):
            if not attribute_name.startswith('_') and hasattr(module, attribute_name):
                attribute = getattr(module, attribute_name)
                if hasattr(attribute, 'rl_text_description'):
                    full_path = f'{attribute.__module__}.{attribute.__name__}'
                    if full_path not in paths_summarized:

                        # print(f'{full_path} ({attribute.rl_text_description}):{attribute.__doc__}')

                        chapter = attribute.rl_text_chapter
                        page = attribute.rl_text_page

                        if chapter not in chapter_page_descriptions:
                            chapter_page_descriptions[chapter] = {}

                        if page not in chapter_page_descriptions[chapter]:
                            chapter_page_descriptions[chapter][page] = []

                        chapter_page_descriptions[chapter][page].append(f'## `{full_path}`\n```\n{attribute.__doc__.strip()}\n```\n')

                        paths_summarized.add(full_path)

        if module_is_pkg:
            summarize(module, chapter_page_descriptions, paths_summarized)


def main():

    chapter_page_descriptions = {}

    # noinspection PyTypeChecker
    summarize(rlai, chapter_page_descriptions)

    meta_md_path = f'{os.path.dirname(__file__)}/../../../README.md'

    # read base readme
    with open('./README_base.md', 'r') as base:
        readme_base = base.read().strip()

    # write markdown file
    with open(meta_md_path, 'w') as meta_md:
        meta_md.write(f'<!--TOC-->\n\n{readme_base}\n\n')
        for chapter in sorted(chapter_page_descriptions):
            meta_md.write(f'# Chapter {chapter}\n')
            for page in sorted(chapter_page_descriptions[chapter]):
                for description in sorted(chapter_page_descriptions[chapter][page]):
                    meta_md.write(description)

    # generate toc
    toc = md_toc.build_toc(meta_md_path)

    # write full markdown file
    with open(meta_md_path, 'r') as meta_md:
        markdown_with_toc = f'# Table of Contents\n{toc}{meta_md.read()}'

    with open(meta_md_path, 'w') as meta_md:
        meta_md.write(markdown_with_toc)


if __name__ == '__main__':
    main()
