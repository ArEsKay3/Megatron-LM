# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU contract tests for the optional frontend formatter hook.

Load the frontend functions without importing the GPU/distributed stack so
these lifecycle and HTTP rejection contracts can also run with --noconftest.
"""

import ast
import asyncio
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVER = ROOT / 'megatron/core/inference/text_generation_server/dynamic_text_gen_server'


def load_function(filename, name, namespace):
    tree = ast.parse((SERVER / filename).read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    node.decorator_list = []
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(SERVER / filename), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('formatter', [None, object()])
def test_formatter_reaches_each_frontend_process(formatter):
    processes = []
    process_type = Mock(
        side_effect=lambda **kwargs: SimpleNamespace(start=Mock(), pid=42, **kwargs)
    )
    worker = Mock()
    start = load_function(
        'text_generation_server.py',
        'start_text_gen_server',
        {
            '_SERVER_PROCESSES': processes,
            'logger': Mock(),
            'mp': SimpleNamespace(Process=process_type),
            '_server_process_worker': worker,
        },
    )
    start(
        'coordinator',
        object(),
        0,
        server_port=9001,
        num_replicas=2,
        hostname='localhost',
        parsers=[],
        response_formatter=formatter,
    )
    assert len(processes) == 2
    for process in processes:
        assert process.target is worker
        assert process.args[-1] is formatter
        process.start.assert_called_once()


def test_legacy_parsers_cannot_silently_override_formatter():
    start = load_function(
        'text_generation_server.py', 'start_text_gen_server', {'_SERVER_PROCESSES': []}
    )
    with pytest.raises(ValueError, match='parsers must be empty'):
        start(
            'coordinator',
            object(),
            0,
            server_port=9001,
            parsers=['old-parser'],
            response_formatter=object(),
        )


def test_invalid_formatter_request_rejected_before_inference():
    class Request:
        async def get_json(self):
            return {'stream': True}

    formatter = Mock()
    formatter.validate_request.side_effect = ValueError('stream=false is required')
    client = Mock()
    handler = load_function(
        'endpoints/chat_completions.py',
        'chat_completions',
        {
            'current_app': SimpleNamespace(
                config={
                    'client': client,
                    'tokenizer': object(),
                    'parsers': [],
                    'response_formatter': formatter,
                }
            ),
            'request': Request(),
            'Response': lambda text, status: (text, status),
        },
    )
    text, status = asyncio.run(handler())
    assert status == 400
    assert 'stream=false' in text
    client.add_request.assert_not_called()


def test_cuda_graph_accepts_derived_inference_contexts(monkeypatch):
    class DynamicContext:
        pass

    class StaticContext:
        pass

    class DerivedContext(DynamicContext):
        pass

    monkeypatch.setitem(
        sys.modules,
        'megatron.core.inference.contexts.dynamic_context',
        SimpleNamespace(DynamicInferenceContext=DynamicContext),
    )
    monkeypatch.setitem(
        sys.modules,
        'megatron.core.inference.contexts.static_context',
        SimpleNamespace(StaticInferenceContext=StaticContext),
    )
    check = load_function(
        ROOT / 'megatron/core/transformer/cuda_graphs.py',
        '_check_supported_type',
        {
            'ArgMetadata': SimpleNamespace,
            'torch': SimpleNamespace(Tensor=type('Tensor', (), {})),
            'dataclass': dataclasses.dataclass,
            'is_dataclass': dataclasses.is_dataclass,
        },
    )
    check(SimpleNamespace(type=DerivedContext, value=DerivedContext()))
    check(SimpleNamespace(type=StaticContext, value=StaticContext()))
    with pytest.raises(AssertionError, match='not supported'):
        check(SimpleNamespace(type=object, value=object()))
