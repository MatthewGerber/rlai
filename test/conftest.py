import os

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def no_show(
        monkeypatch  # noqa
):
    """
    Do not show plots when running tests at the top-level `test` scope, as this eats up memory when showing plots in the
    PyCharm SciView window and is generally annoying and uninformative.

    :param monkeypatch: Patch.
    """

    if os.environ.get('SHOW_PLOTS', 'True') == 'False':
        monkeypatch.setattr(plt, 'show', lambda block=False: plt.close('all'))
        monkeypatch.setattr(plt.Figure, 'show', lambda block=False: plt.close('all'))
