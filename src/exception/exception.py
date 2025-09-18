"""
==================================================================
CUSTOM EXCEPTION
------------------------------------------------------------------
A custom exception class for the forecasting pipeline.

✔️ Provides detailed error messages including file name and line number
✔️ Automatically logs errors to a central log file
✔️ Can be used across all modules in the project (loader, feature engineering,
   model fitting, prediction, evaluation)
✔️ Simplifies debugging and monitoring in multi-series forecasting pipelines

⚠️ This class is not for business logic errors — only for capturing and 
   reporting system or code exceptions.
==================================================================
"""

import sys
import logging
from pathlib import Path

# Thiết lập logging cơ bản
log_path = Path("logs")
log_path.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_path / "pipeline_errors.log",
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.ERROR
)

# 1.
def detailed_error_msg(error, error_details: sys):
    """
    Generate a detailed error message including file name, line number, and error message.

    Args:
        error: The original exception.
        error_details (sys): System information about the exception.

    Returns:
        str: A detailed error message.
    """
    _, _, exception_traceback = error_details.exc_info()
    file_name = exception_traceback.tb_frame.f_code.co_filename
    line_number = exception_traceback.tb_lineno
    return f"An error occurred in file [{file_name}] line [{line_number}]: {str(error)}"

class CustomException(Exception):
    """
    Custom exception class that logs the error and provides detailed traceback info.
    """
    def __init__(self, error: Exception, error_details: sys):
        """
        Initialize the CustomException instance.

        Args:
            error (Exception): The original exception.
            error_details (sys): System info for traceback.
        """
        self.detailed_error_message = detailed_error_msg(error, error_details)
        super().__init__(self.detailed_error_message)

        # Ghi lỗi vào log file tự động
        logging.error(self.detailed_error_message)

    def __str__(self):
        return self.detailed_error_message
