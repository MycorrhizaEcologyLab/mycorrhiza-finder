# TODO replace with loguru

# AMFinder - log.py
#
# MIT License
# Copyright (c) 2021 Edouard Evangelisti, Carl Turner
#               2024-2025 Royal Botanic Gardens, Kew
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

"""
Log functions.

Displays message, warnings, errors, and progress bars.

Constants
-----------
ERR_INVALID_MODEL - Wrong model.
ERR_NO_PRETRAINED_MODEL - A pre-trained model is required but not provided.
ERR_INVALID_MODEL_SHAPE - Wrong model input shape.
ERR_INVALID_ANNOTATION_LEVEL -
ERR_MISSING_ARCHIVE - Cannot find the zip archive associated with an image.
ERR_MISSING_ANNOTATIONS - The given archive lacks stage 1 annotations.

Functions
-----------
:function info: prints a message.
:function warning: prints a warning message.
:function error: prints an error message, and closes the application.
:function progress_bar: displays a progress bar.

"""

import datetime
import sys
from typing import Any

from loguru import logger as logger

ERR_NO_DATA = 10
ERR_INVALID_DATA = 11
ERR_NO_PRETRAINED_MODEL = 20
ERR_INVALID_MODEL_SHAPE = 21
ERR_INVALID_ANNOTATION_LEVEL = 22
ERR_MISSING_ARCHIVE = 30
ERR_MISSING_ANNOTATIONS = 32
ERR_INVALID_MODEL = 40
ERR_NO_DATABASE_CONNECTION = 42

logger.remove(0)
# lg.add(sys.stdout, format="{time:MMMM D, YYYY - HH:mm:ss} | {level}")


def formatter(record):
    if record["level"].no == 10:
        return "{time:MMMM D, YYYY - HH:mm:ss} | {message}\n"
    else:
        return "{time:MMMM D, YYYY - HH:mm:ss} | {level} | {message}\n"


logger.add(f"{datetime.date.today()}.log", format=formatter, level=0)
logger.add(sys.stderr, format=formatter, level=0)

# log.add(sys.stderr, level="DEBUG", format="{time:MMMM D, YYYY - HH:mm:ss} | {message}")
# log.add(sys.stderr, format="{time:MMMM D, YYYY - HH:mm:ss} | {level} | {message}")


# def invite() -> str:
#     """
#     Command-line invite.
#     """
#     return f"[{datetime.datetime.now().strftime('%H:%M:%S')}]"


# def text(message: str, **kwargs: Any) -> None:
#     """
#     Prints an message on standard output.

#     :param message: The message to be printed.
#     :param ident: Indentation level (defaults to 0).
#     :param kwargs: Any relevant keyword argument.
#     """

#     print(f"{invite()} {message}.", **kwargs)


# def info(message: str, **kwargs: Any) -> None:
#     """
#     Prints an message on standard output.

#     :param message: The message to be printed.
#     :param ident: Indentation level (defaults to 0).
#     :param kwargs: Any relevant keyword argument.
#     """

#     print(f"{invite()} INFO: {message}.", **kwargs)


# def warning(message: str, **kwargs: Any) -> None:
#     """
#     Prints a warning message on standard error.

#     :param message: The message to be printed.
#     :param ident: Indentation level (defaults to 0).
#     :param kwargs: Any relevant keyword argument.
#     """

#     print(f"{invite()} WARNING: {message}.", file=sys.stderr, **kwargs)


# def error(message: str, exit_code: int, **kwargs: Any) -> None:
#     """
#     Prints an error message on standard error and quits.

#     :param message: The message to be printed.
#     :param exit_code: The exit code to return when closing the application.
#     :param ident: Indentation level (defaults to 0).
#     :param kwargs: Any relevant keyword argument.
#     """

#     print(f"{invite()} ERROR: {message}.", file=sys.stderr, **kwargs)

#     if exit_code is not None and exit_code != 0:
#         sys.exit(exit_code)


# def progress_bar(iteration: int, total: int, indent: int = 0) -> None:
#     """
#     Displays a progress bar.

#     :param iteration: Current iteration value.
#     :param total: Total iteration count (to calculate percentages).
#     :param indent: Indentation level (defaults to 0).
#     """

#     if total > 0:
#         percent = 100.0 * iteration / float(total)

#         completed = round(50.0 * iteration / total)
#         remaining = 50 - completed

#         bar = "█" * completed + "-" * remaining

#         print(" " * 4 * indent + f"- processing |{bar}| {percent:.1f}%", end="\r")

#         if iteration == total:
#             print()  # newline
