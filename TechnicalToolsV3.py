"""
This is the Technical tools python library created for general purpose stuff
just so you know I have no idea what half of this means so don't ask
"""

from random import randint, choice
from inspect import currentframe, getframeinfo
import fractions
import os


class LogConfig:
    show_filename = False
    show_line_number = False

    color_codes = {
        "black": 30,
        "grey": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "lgrey": 37,
        "dgrey": 90,
        "lred": 91,
        "lgreen": 92,
        "lyellow": 93,
        "lblue": 94,
        "lmagenta": 95,
        "lcyan": 96,
        "white": 97,
    }

    mode_to_config = {
        "info": ["INFO", "blue"],
        "error": ["ERROR", "lred"],
        "warning": ["WARNING", "lyellow"],
        "success": ["SUCCESS", "lgreen"],
        "result": ["RESULT", "magenta"],
    }

    @classmethod
    def add_config(cls, mode, logtitle, color):
        if color not in cls.color_codes:
            log("Color not found in config.", mode="error")

        cls.mode_to_config[mode] = [logtitle, color]

    def __init__(self, mode):
        if mode in self.mode_to_config:
            mode_info = self.mode_to_config[mode]
        else:
            log("Specified mode not in config, defaulting to info.", mode="warning")
            mode_info = self.mode_to_config["info"]
        self.logtitle = mode_info[0]
        self.color = mode_info[1]


def log(message: str = "", mode: str = "", var: any = None, logtitle: str = "info", color: str = "blue", include_var_formatting: bool = True):
    """
    Printing but with color!
    :param include_var_formatting:
    :param message:
    :param mode:
    :param var:
    :param logtitle:
    :param color:
    """

    #  Setup current mode
    if mode:
        current_mode = LogConfig(mode=mode)
        logtitle = current_mode.logtitle
        color = current_mode.color

    # Format the variable
    if var is None:
        var = ""
    elif isinstance(var, dict) or isinstance(var, list) or isinstance(var, tuple):
        var = str(var)
    elif include_var_formatting:
        var = '|' + str(var) + '|'
    else:
        var = '\n' + str(var)

    try:
        color = LogConfig.color_codes[color]
    except KeyError:
        log("Color not found, defaulting to cyan", mode="warning")
        color = 36

    filename_txt = ""
    linenumber_txt = ""
    logtitle_txt = ""
    dash_txt = " - "
    null_escape = '\033[0m'
    color_escape = f"\033[0;{color}m"

    if LogConfig.show_filename:
        if os.name == 'nt':  # Windows specifically
            filename = getframeinfo(currentframe().f_back).filename.split('\\')[-1]
        else:
            filename = getframeinfo(currentframe().f_back).filename.split('/')[-1]
        filename_txt = f" in \033[4;{color}m{filename}{color_escape}"

    if LogConfig.show_line_number:
        linenumber = currentframe().f_back.f_lineno
        linenumber_txt = f":[{linenumber}]"

    if logtitle:
        logtitle_txt = f"\033[1;{color}m[{logtitle.upper()}]{color_escape}"

    if not message:
        dash_txt = ""

    print_message = f'{logtitle_txt}{filename_txt}{linenumber_txt}{dash_txt}{message} {var}{null_escape}'
    print(print_message)


def time_convert(seconds):
    """
    turns absurd amounts of seconds into understandable text
    :param seconds:
    :return:
    """

    year = int(31536000 * 1e9)
    month = int(2592000 * 1e9)
    week = int(604800 * 1e9)
    day = int(86400 * 1e9)
    hour = int(3600 * 1e9)
    minute = int(60 * 1e9)
    second = int(1 * 1e9)
    millisecond = 1000000
    microsecond = 1000
    nanosecond = 1
    seconds *= int(1e9)

    return_text = ''

    amount = seconds // year
    if amount > 0:
        s = ""
        if amount > 1: s = "s"
        return_text = return_text + f' {int(amount)} Year{s}'
        seconds -= year * amount

    amount = seconds // month
    if amount > 0:
        s = ""
        if amount > 1: s = "s"
        return_text = return_text + f' {int(amount)} Month{s}'
        seconds -= month * amount

    amount = seconds // week
    if amount > 0:
        s = ""
        if amount > 1: s = "s"
        return_text = return_text + f' {int(amount)} Week{s}'
        seconds -= week * amount

    amount = seconds // day
    if amount > 0:
        s = ""
        if amount > 1: s = "s"
        return_text = return_text + f' {int(amount)} Day{s}'
        seconds -= day * amount

    amount = seconds // hour
    if amount > 0:
        s = ""
        if amount > 1: s = "s"
        return_text = return_text + f' {int(amount)} Hour{s}'
        seconds -= hour * amount

    amount = seconds // minute
    if amount > 0:
        s = ""
        if amount > 1: s = "s"
        return_text = return_text + f' {int(amount)} Minute{s}'
        seconds -= minute * amount

    amount = seconds // second
    if amount > 0:
        s = ""
        if amount > 1: s = "s"
        return_text = return_text + f' {int(amount)} Second{s}'
        seconds -= second * amount

    amount = seconds // millisecond
    if amount > 0:
        s = ""
        if amount > 1: s = "s"
        return_text = return_text + f' {int(amount)} MilliSecond{s}'
        seconds -= millisecond * amount

    amount = seconds // microsecond
    if amount > 0:
        s = ""
        if amount > 1: s = "s"
        return_text = return_text + f' {int(amount)} MicroSecond{s}'
        seconds -= microsecond * amount

    amount = seconds // nanosecond
    if amount > 0:
        s = ""
        if amount > 1: s = "s"
        return_text = return_text + f' {int(amount)} NanoSecond{s}'
        seconds -= nanosecond * amount

    return return_text + ' '


def multi_delete(arr, indexes):
    """
    Deletes all indexes in arr
    :param arr:
    :param indexes:
    """
    indexes = set(indexes)
    indexes = sorted(set(indexes), reverse=True)
    for index in indexes:
        del arr[index]
    return arr


class CNumber:
    """
    CNumber -> Cached number.
    """

    # +5-3*67**1/2+9/45 -> float

    def __init__(self, val):
        raise NotImplementedError


if __name__ == '__main__':
    log("Completed in", mode="success", var=time_convert(1002004))
