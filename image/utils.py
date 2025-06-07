import json

import requests

from configs.env_config import settings
from shared.util import logging_utils
from shared.util.logging_utils import logger

TIME_OUT_SETTING = (30.0, 30.0)

BACKEND_ENDPOINT = settings.BACKEND_ENDPOINT


def request_backend(path, method, authentication, data={}, params={}):
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    if method not in methods:
        return f"Invalid method {method}"

    headers = {"Content-Type": "application/json", "Authentication": authentication}
    url = f"{BACKEND_ENDPOINT}{path}"
    res, err = _request(url, method, headers, data, params)
    if err:
        return None, err

    if res.status_code != 200:
        logger.error(
            f"backendへのリクエストエラー: Response status: {res.status_code} Response content: {res.text}"
        )
        return None, Exception(f"Response status {res.status_code}")

    return res, err


def _request(url, method, headers, data, params):
    try:
        if method == "GET":
            res = requests.get(
                url, headers=headers, params=params, timeout=TIME_OUT_SETTING
            )
        elif method == "POST":
            res = requests.post(
                url, headers=headers, data=json.dumps(data), timeout=TIME_OUT_SETTING
            )
        elif method == "PUT":
            res = requests.put(
                url, headers=headers, data=json.dumps(data), timeout=TIME_OUT_SETTING
            )
        elif method == "PATCH":
            res = requests.patch(
                url, headers=headers, data=json.dumps(data), timeout=TIME_OUT_SETTING
            )
        elif method == "DELETE":
            res = requests.delete(
                url, headers=headers, data=json.dumps(data), timeout=TIME_OUT_SETTING
            )
    except Exception as e:
        logger.error(f"backendへのリクエストエラー {logging_utils.get_formatted_traceback()}")
        return None, e
    return res, None
