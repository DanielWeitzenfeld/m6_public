import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


def to_percent_2fig(y, position):
    s = "%1.2f" % (100 * y)
    return s + "%"


percent_formatter_2fig = FuncFormatter(to_percent_2fig)


def to_percent(y, position):
    s = "%1.0f" % (100 * y)
    return s + "%"


percent_formatter = FuncFormatter(to_percent)
