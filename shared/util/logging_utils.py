import functools
import inspect
import json
import logging
import os
import time
import traceback
from contextvars import ContextVar
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.routing import APIRoute

from configs.env_config import settings

ENV = settings.ENVIRONMENT


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "%(levelname)s: [%(asctime)s][%(pathname)s][%(funcName)s,%(lineno)d] %(message)s"
    )
)
logger.addHandler(handler)

company_id_context: ContextVar[Optional[str]] = ContextVar("company_id", default=None)


def get_formatted_traceback():
    return "| {}".format(traceback.format_exc().replace("\n", "\\n").replace('"', "'"))


def truncate_arguments(arguments, max_size_kb=150):
    max_size_bytes = max_size_kb * 1024  # 150KB
    if len(arguments.encode("utf-8")) > max_size_bytes:
        truncated_arguments = arguments.encode("utf-8")[:max_size_bytes].decode(
            "utf-8", errors="ignore"
        )
        return truncated_arguments + "…[サイズが原因で切り捨てられた引数]"
    return arguments


def convert_function_arguments_to_string_for_logging(*args, **kwargs):
    out = ""
    for arg in args:
        if isinstance(arg, list) and len(arg) >= 100:
            out += f"[{arg[0]}, {arg[1]}, ..., {arg[-1]}] (length={len(arg)}), "
        else:
            out += f"{repr(arg)}, "
    for key, value in kwargs.items():
        if key == "authentication":
            out += f"{key}=masked, "
        elif isinstance(value, list) and len(value) >= 100:
            out += f"{key}=[{value[0]}, {value[1]}, ..., {value[-1]}] (length={len(value)}), "
        else:
            out += f"{key}={repr(value)}, "
    return out[:-2]


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        company_id = company_id_context.get()
        current_frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(current_frame, 2)

        arguments = convert_function_arguments_to_string_for_logging(*args, **kwargs)
        arguments = truncate_arguments(arguments)

        # 関数のファイル名と行番号を取得
        file_path = caller_frame[1].filename
        line_number = caller_frame[1].lineno

        log_data_tmpl = {
            "env": ENV,
            "company_id": company_id,
            "file_path": file_path,
            "function_name": func.__name__,
            "line_number": line_number,
        }

        log_data = log_data_tmpl.copy()
        log_data["message"] = f"Function '{func.__name__}'({arguments}) started"
        logger.info(json.dumps(log_data))
            

        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            log_data = log_data_tmpl.copy()
            log_data["message"] = (
                f"Function '{func.__name__}'executed in {elapsed_time:.6f} seconds"
            )
            logger.info(json.dumps(log_data))
                

            return result
        except Exception as e:
            log_data = log_data_tmpl.copy()
            log_data["message"] = (
                f"Error occured in function '{func.__name__}({arguments})'"
            )
            log_data["error_message"] = str(e)
            log_data["traceback"] = get_formatted_traceback()

            logger.error(json.dumps(log_data))
            raise

    return wrapper


class LoggingContextRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            log_data = {
                "request_headers": self._get_request_header(request),
                "remote_addr": request.client.host,
                "request_uri": request.url.path,
                "request_method": request.method,
                "request_body": await self._get_request_body(request),
            }

            try:
                response = await original_route_handler(request)
                self._add_response_info_to_log_data(response, log_data)
                logger.info(json.dumps(log_data))
            except Exception as e:
                self._add_error_to_log_data(e, log_data)
                logger.error(json.dumps(log_data))
                raise

            return response

        return custom_route_handler

    async def _get_request_body(self, request: Request) -> str:
        try:
            body = await request.body()
            return body.decode("utf-8") if body else ""
        except Exception:
            return "Error in decoding request body"

    def _add_response_info_to_log_data(self, response: Response, log_data: dict):
        if response:
            log_data.update(
                {
                    "response_status": response.status_code,
                    "response_headers": dict(response.headers),
                    "response_body": (
                        (response.body.decode("utf-8")[:100] + "...")
                        if len(response.body.decode("utf-8")) > 100
                        else response.body.decode("utf-8")
                    ),
                }
            )

    def _add_error_to_log_data(self, error: Exception, log_data: dict):
        log_data.update({"error": str(error), "traceback": traceback.format_exc()})

    def _get_request_header(self, request: Request) -> dict:
        headers = dict(request.headers)

        if "authentication" in headers:
            auth_type, _ = headers["authentication"].split(" ")
            if auth_type.lower() == "bearer":
                headers["authentication"] = f"{auth_type} `masked`"

        return headers