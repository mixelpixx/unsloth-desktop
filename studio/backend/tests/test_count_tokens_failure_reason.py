# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Why /chat/count_tokens answered 503 has to reach the server log.

Field log: two 503s right after a load, nothing else. The model's chat template raises
"No user query found" for a prompt with no user turn yet, /apply-template answers with
that error, and the strict count turned it into a generic RuntimeError that the route's
catch-all turned into a generic 503. The client still gets the same 503; the log now says
why.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import core.inference.llama_cpp as llama_cpp


class _Resp:
    def __init__(self, status_code, payload = None, text = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _client_answering(apply_template_resp):
    class _Client:
        def __init__(self, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json = None):
            if url.endswith("/apply-template"):
                if isinstance(apply_template_resp, BaseException):
                    raise apply_template_resp
                return apply_template_resp
            return _Resp(200, {"tokens": [1, 2, 3]})

    return _Client


class _Backend(llama_cpp.LlamaCppBackend):
    is_loaded = True
    base_url = "http://127.0.0.1:8080"
    _auth_headers: dict = {}


def _count(monkeypatch, resp, *, strict = True):
    monkeypatch.setattr(llama_cpp.httpx, "Client", _client_answering(resp))
    return _Backend.__new__(_Backend).count_chat_tokens(
        [{"role": "system", "content": "You are helpful."}], None, None, strict = strict
    )


def test_the_template_error_is_carried_as_the_reason(monkeypatch):
    resp = _Resp(
        400,
        {"error": {"code": 400, "message": "No user query found in messages.", "type": "x"}},
    )
    with pytest.raises(RuntimeError) as err:
        _count(monkeypatch, resp)
    # The message callers may echo to an API client is unchanged.
    assert str(err.value) == "llama-server could not render the chat template"
    assert err.value.reason == "/apply-template HTTP 400: No user query found in messages."


def test_a_transport_error_is_carried_too(monkeypatch):
    with pytest.raises(RuntimeError) as err:
        _count(monkeypatch, ConnectionError("refused"))
    assert err.value.reason == "ConnectionError: refused"


def test_non_strict_callers_still_get_the_text_fallback(monkeypatch):
    assert _count(monkeypatch, _Resp(400, {"error": {"message": "x"}}), strict = False) == 3


@pytest.mark.parametrize(
    "resp,expected",
    [
        (_Resp(500, {"error": "plain string error"}), "plain string error"),
        (_Resp(500, None, "  raw\n  body  "), "raw body"),
        (_Resp(500, None, ""), "no detail"),
        (_Resp(500, {"error": {"message": "y" * 400}}), "y" * 300 + "..."),
    ],
)
def test_failure_detail_is_one_bounded_line(resp, expected):
    assert llama_cpp._count_failure_detail(resp) == expected


def test_the_route_logs_the_reason_before_the_503():
    import routes.inference as routes

    source = inspect.getsource(routes.chat_count_tokens)
    handler = source[source.index("except CountAborted:") :]
    catch_all = handler[handler.index("except Exception as exc:") :]
    assert catch_all.index("logger.info(") < catch_all.index("raise HTTPException(")
    assert 'getattr(exc, "reason", None)' in catch_all
    assert "Unable to count tokens with the loaded model tokenizer." in catch_all
