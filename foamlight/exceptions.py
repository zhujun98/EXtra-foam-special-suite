"""
Distributed under the terms of the MIT License.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""


class ProcessingError(Exception):
    """Base Exception for non-fatal errors in pipeline.

    The error must not be fatal and the rest of the data processing pipeline
    can still resume.
    """
    pass
