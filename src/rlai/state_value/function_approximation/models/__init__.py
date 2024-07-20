from abc import ABC
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rlai.models import FunctionApproximationModel


class StateFunctionApproximationModel(FunctionApproximationModel, ABC):
    """
    Base class for models that approximate state-action value functions.
    """

    def plot(
            self,
            render: bool,
            pdf: Optional[PdfPages]
    ) -> Optional[plt.Figure]:
        """
        Plot the model. If called from the main thread and render is True, then a new plot will be generated. If called
        from a background thread, then the data used by the plot will be updated but a plot will not be generated or
        updated. This supports a pattern in which a background thread generates new plot data, and a UI thread (e.g., in
        a Jupyter notebook) periodically calls `update_plot` to redraw the plot with the latest data.

        :param render: Whether to render the plot data. If False, then plot data will be updated but nothing will
        be shown.
        :param pdf: PDF to plot to, or None to show directly.
        :return: Matplotlib figure, if one was generated and not plotting to PDF.
        """
