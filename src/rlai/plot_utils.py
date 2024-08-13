from enum import Enum, auto
from typing import List, Optional, Dict

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from rlai.docs import rl_text


# noinspection PyArgumentList
class ScatterPlotPosition(Enum):
    """
    Scatter-plot positions.
    """

    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()


@rl_text('Diagnostics', page=1)
class ScatterPlot:
    """
    General-purpose scatter plot that supports real-time plot updates.
    """

    next_position = ScatterPlotPosition.TOP_LEFT

    def __init__(
            self,
            title: str,
            x_tick_labels: List[str],
            position: Optional[ScatterPlotPosition]
    ):
        """
        Initialize the scatter plot.

        :param title: Title.
        :param x_tick_labels: Labels for the x-axis ticks.
        :param position: Position, or None to place the plot automatically on the screen.
        """

        if position is None:
            position = ScatterPlot.next_position
            if position == ScatterPlotPosition.TOP_LEFT:
                ScatterPlot.next_position = ScatterPlotPosition.TOP_RIGHT
            elif position == ScatterPlotPosition.TOP_RIGHT:
                ScatterPlot.next_position = ScatterPlotPosition.BOTTOM_RIGHT
            elif position == ScatterPlotPosition.BOTTOM_RIGHT:
                ScatterPlot.next_position = ScatterPlotPosition.BOTTOM_LEFT
            else:
                ScatterPlot.next_position = ScatterPlotPosition.TOP_LEFT

        self.title = title
        self.x_tick_labels = x_tick_labels
        self.position = position

        self.plot_layout = pg.GraphicsLayoutWidget(show=True, title=title)
        plot_x_axis = pg.AxisItem(orientation='bottom')
        plot_x_axis.setTicks([list(enumerate(self.x_tick_labels))])
        self.plot_widget = self.plot_layout.addPlot(axisItems={'bottom': plot_x_axis})
        self.plot_item = None
        self.plot_x_vals = list(range(len(x_tick_labels)))
        self.plot_max_abs_y = None

        self.set_position(self.position)

    def update(
            self,
            y_values: np.ndarray
    ):
        """
        Update the scatter plot.

        :param y_values: New y values.
        """

        if np.isinf(y_values).any() or np.isnan(y_values).any():
            return

        # expand y range if needed. never shrink it. this helps to keep the visual interpretable.
        max_abs_y = np.abs(y_values).max()
        if self.plot_max_abs_y is None or max_abs_y > self.plot_max_abs_y:
            self.plot_max_abs_y = max_abs_y
            assert self.plot_max_abs_y is not None
            self.plot_widget.setYRange(-self.plot_max_abs_y, self.plot_max_abs_y)

        # create initial plot item if we don't have one
        if self.plot_item is None:
            self.plot_item = self.plot_widget.plot(self.plot_x_vals, y_values, pen=pg.mkPen(None), symbol='o')

        # update data in plot item if we already have one
        else:
            self.plot_item.setData(self.plot_x_vals, y_values)

    def reset_y_range(
            self
    ):
        """
        Reset the y-axis range, so that the next call to `update` will determine it.
        """

        self.plot_max_abs_y = None

    def move_to_top_left(self):
        """
        Move the scatter plot to the top-left corner of the screen.
        """

        screen = QApplication.primaryScreen()
        assert screen is not None

        top_left_point = screen.availableGeometry().topLeft()
        plot_layout_geometry = self.plot_layout.frameGeometry()
        plot_layout_geometry.moveTopLeft(top_left_point)
        self.plot_layout.move(plot_layout_geometry.topLeft())

    def move_to_top_right(self):
        """
        Move the scatter plot to the top-right corner of the screen.
        """

        screen = QApplication.primaryScreen()
        assert screen is not None

        top_right_point = screen.availableGeometry().topRight()
        plot_layout_geometry = self.plot_layout.frameGeometry()
        plot_layout_geometry.moveTopRight(top_right_point)
        self.plot_layout.move(plot_layout_geometry.topLeft())

    def move_to_bottom_left(self):
        """
        Move the scatter plot to the bottom-left corner of the screen.
        """

        screen = QApplication.primaryScreen()
        assert screen is not None

        bottom_left_point = screen.availableGeometry().bottomLeft()
        plot_layout_geometry = self.plot_layout.frameGeometry()
        plot_layout_geometry.moveBottomLeft(bottom_left_point)
        self.plot_layout.move(plot_layout_geometry.topLeft())

    def move_to_bottom_right(self):
        """
        Move the scatter plot to the bottom-right corner of the screen.
        """

        screen = QApplication.primaryScreen()
        assert screen is not None

        bottom_right_point = screen.availableGeometry().bottomRight()
        plot_layout_geometry = self.plot_layout.frameGeometry()
        plot_layout_geometry.moveBottomRight(bottom_right_point)
        self.plot_layout.move(plot_layout_geometry.topLeft())

    def set_position(
            self,
            position: ScatterPlotPosition
    ):
        """
        Set the position of the scatter plot on the screen.

        :param position: Position.
        """

        self.position = position

        if self.position == ScatterPlotPosition.TOP_LEFT:
            self.move_to_top_left()
        elif self.position == ScatterPlotPosition.TOP_RIGHT:
            self.move_to_top_right()
        elif self.position == ScatterPlotPosition.BOTTOM_LEFT:
            self.move_to_bottom_left()
        else:
            self.move_to_bottom_right()

    def close(
            self
    ):
        """
        Close the scatter plot.
        """

        self.plot_layout.close()

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: State dictionary.
        """

        state = dict(self.__dict__)

        state['plot_layout'] = None
        state['plot_widget'] = None
        state['plot_item'] = None

        return state

    def __setstate__(
            self,
            state: Dict
    ):
        """
        Set the state dictionary.

        :param state: State dictionary.
        """

        self.__dict__ = state

        self.plot_layout = pg.GraphicsLayoutWidget(show=True, title=self.title)
        plot_x_axis = pg.AxisItem(orientation='bottom')
        plot_x_axis.setTicks([list(enumerate(self.x_tick_labels))])
        self.plot_widget = self.plot_layout.addPlot(axisItems={'bottom': plot_x_axis})
        self.set_position(self.position)
