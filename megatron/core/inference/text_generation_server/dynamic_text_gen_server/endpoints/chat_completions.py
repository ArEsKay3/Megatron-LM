# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import itertools
import json
import logging
import math
import os
import time
import traceback
import uuid
import warnings
from functools import partial

import torch

from megatron.core.inference.inference_request import (
    compute_block_hashes_batched,
    unwrap_serialized_tensors,
)
from megatron.core.inference.config import routes_on_prefix
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.tokenizers.text.parsers import PARSER_MAPPING

from ..incremental_detokenizer import HuggingFaceFastIncrementalDetokenizer
from ..openai_streaming import openai_stream

logger = logging.getLogger(__name__)

# --- [FE-EV] per-event request tracing -----------------------------------
# Every counter change emits its own line with a wall-clock stamp instead of a
# periodic aggregate. The previous aggregate was sampled only when a request
# completed, which biased it toward busy moments and made it non-comparable
# with the coordinator's counters; it also folded tokenizer queue wait and
# tokenizer service time into one number. Events are emitted raw and
# un-aggregated so they can be joined on `ts` across processes.
#
# Event vocabulary (one line each):
#   recv     request arrived at the blueprint
#   tok_enq  handed to the single-worker tokenize executor  (queue wait starts)
#   tok_svc  emitted ON the executor thread: SERVICE time only, plus ntok/nmsg
#   tok_deq  executor future resolved                        (queue wait ends)
#   gen_beg  submitted to the engine, awaiting gather
#   gen_end  all n engine replies received
#   done     response finished (every exit path, incl. errors)
# h/t/g are the handler/tokenize/generate depths *after* the change.
# Set NRL_FE_EVENTS=0 to silence.
_FE_EV = os.environ.get("NRL_FE_EVENTS", "1") not in ("0", "false", "")
_fe_gauge = {"handler": 0, "tokenize": 0, "generate": 0}
_fe_rid = itertools.count(1)

# [FE-REQ] One-line summary per inbound request. Expensive at scale: 31k lines
# / 24 MB on one SWE run. Default OFF; set NRL_FE_REQ_SUMMARY=1 to enable.
_FE_REQ_SUMMARY = os.environ.get("NRL_FE_REQ_SUMMARY", "0") not in ("0", "false", "")


def _ev(ev, rid, **kv):
    """Emit one event line. print(), not logging: the frontend replica
    processes run with the root logger above INFO, so logging.info is dropped."""
    if not _FE_EV:
        return
    extra = "".join(f" {k}={v}" for k, v in kv.items())
    print(
        f"[FE-EV] ts={time.time():.4f} pid={os.getpid()} rid={rid} ev={ev}"
        f" h={_fe_gauge['handler']} t={_fe_gauge['tokenize']} g={_fe_gauge['generate']}"
        f"{extra}",
        flush=True,
    )


# pylint: disable=line-too-long

_TOKEN_ID_FIELDS_TO_REDACT = {
    "prompt_tokens",
    "remaining_prompt_tokens",
    "generated_tokens",
    "prompt_token_ids",
    "generation_token_ids",
}

_INDEX_FIELDS_TO_REDACT = {"routing_indices", "moe_topk_indices", "prompt_moe_topk_indices"}

_HASH_FIELDS_TO_REDACT = {"precomputed_block_hashes"}

_NUMERIC_SERIES_FIELDS_TO_REDACT = {"tpot"}


def _is_int_list_like(value):
    """Return True for integer lists, including nested integer lists."""
    if not isinstance(value, list):
        return False
    return all(isinstance(item, int) or _is_int_list_like(item) for item in value)


def _is_numeric_list_like(value):
    """Return True for numeric lists, including nested numeric lists."""
    if not isinstance(value, list):
        return False
    return all(isinstance(item, (int, float)) or _is_numeric_list_like(item) for item in value)


def _redact_token_id_lists_for_logging(value):
    """Redact verbose token-id arrays from logs."""
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if (
                key in _TOKEN_ID_FIELDS_TO_REDACT
                or key in _INDEX_FIELDS_TO_REDACT
                or key in _HASH_FIELDS_TO_REDACT
                or key.endswith("_token_ids")
                or key.endswith("_topk_indices")
                or key.endswith("_hashes")
            ) and _is_int_list_like(item):
                redacted[key] = "...truncated..."
            elif key in _NUMERIC_SERIES_FIELDS_TO_REDACT and _is_numeric_list_like(item):
                redacted[key] = "...truncated..."
            else:
                redacted[key] = _redact_token_id_lists_for_logging(item)
        return redacted
    if isinstance(value, list):
        return [_redact_token_id_lists_for_logging(item) for item in value]
    return value


def _get_field(obj, key, default=None):
    """Read a field from dict-like or object-like values."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_non_none(obj, key, default):
    """Returns the value from the object or default if the key is missing or None."""
    val = obj.get(key)
    return default if val is None else val


def _try_parse_jsonish(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return value


def _extract_declared_types(schema):
    """Recursively extract declared JSON-schema type names."""
    declared = set()
    if not isinstance(schema, dict):
        return declared

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        declared.add(schema_type.strip().lower())
    elif isinstance(schema_type, list):
        for item in schema_type:
            if isinstance(item, str):
                declared.add(item.strip().lower())

    for combinator in ("anyOf", "oneOf", "allOf"):
        options = schema.get(combinator)
        if isinstance(options, list):
            for option in options:
                declared.update(_extract_declared_types(option))
    return declared


def _get_tool_argument_schemas(tools):
    """Build function-name to argument-schema mapping from request tools."""
    schemas = {}
    if not isinstance(tools, list):
        return schemas

    for tool in tools:
        function = _get_field(tool, "function", {}) or {}
        function_name = _get_field(function, "name")
        params = _get_field(function, "parameters", {})
        if not isinstance(function_name, str) or not isinstance(params, dict):
            continue
        if isinstance(params.get("properties"), dict):
            schemas[function_name] = params.get("properties")
        else:
            schemas[function_name] = params
    return schemas


def _normalize_structured_tool_arguments(arguments, function_name, tool_argument_schemas):
    """Coerce structured (array/object) args from JSON strings to native types."""
    if not isinstance(arguments, dict):
        return arguments

    function_schema = tool_argument_schemas.get(function_name, {})
    if not isinstance(function_schema, dict):
        return arguments

    normalized = dict(arguments)
    for key in normalized:
        param_schema = function_schema.get(key)
        declared_types = _extract_declared_types(param_schema)
        if not (declared_types & {"array", "arr", "object", "dict", "list"}):
            continue
        parsed = _try_parse_jsonish(normalized[key])
        if isinstance(parsed, (dict, list)):
            normalized[key] = parsed
    return normalized


def _normalize_tool_calls(tool_calls, tools=None):
    """Normalize tool calls to OpenAI-compatible JSON primitives."""
    tool_argument_schemas = _get_tool_argument_schemas(tools)
    normalized = []
    for call in tool_calls or []:
        fn = _get_field(call, "function", {}) or {}
        fn_name = _get_field(fn, "name")
        fn_args = _get_field(fn, "arguments", "")
        if fn_name is None:
            continue
        if isinstance(fn_args, str):
            try:
                parsed_args = json.loads(fn_args)
            except (TypeError, ValueError):
                parsed_args = None
            if isinstance(parsed_args, dict):
                fn_args = json.dumps(
                    _normalize_structured_tool_arguments(
                        parsed_args, fn_name, tool_argument_schemas
                    ),
                    ensure_ascii=False,
                )
        elif isinstance(fn_args, dict):
            fn_args = json.dumps(
                _normalize_structured_tool_arguments(fn_args, fn_name, tool_argument_schemas),
                ensure_ascii=False,
            )
        else:
            try:
                fn_args = json.dumps(fn_args, ensure_ascii=False)
            except TypeError:
                fn_args = str(fn_args)
        normalized.append(
            {
                "id": str(_get_field(call, "id", f"call_{uuid.uuid4().hex[:24]}")),
                "type": "function",
                "function": {"name": str(fn_name), "arguments": fn_args},
            }
        )
    return normalized


def _maybe_filter_parallel_tool_calls(tool_calls, parallel_tool_calls):
    """Filter to first tool call only when parallel_tool_calls is False.

    Matches vLLM's maybe_filter_parallel_tool_calls behavior.
    """
    if parallel_tool_calls:
        return tool_calls
    if tool_calls:
        return tool_calls[:1]
    return tool_calls


def _coerce_arguments_mapping(arguments):
    """Coerce function.arguments to a mapping for HF/Jinja chat templates.

    Examples:
    - {"x": 1} -> {"x": 1}
    - '{"x": 1}' -> {"x": 1}
    - "[1, 2]" -> {}  # JSON parses, but not a mapping
    - "not-json" -> {}
    - None -> {}
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _sanitize_messages_for_template(messages):
    """Prepare messages so tokenizer chat templates can safely consume them.

    This only normalizes tool-call argument payloads inside each message:
    - messages[*].tool_calls[*].function.arguments is coerced to a dict.

    Example transformation:
    Input:
      [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{\"x\": 1}"}}]}]
    Output:
      [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]}]

    Another example:
    - arguments: "[1,2,3]" -> arguments: {}
    """
    if not isinstance(messages, list):
        return messages
    sanitized = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        msg_copy = dict(message)
        content = msg_copy.get("content")
        # OpenAI-style multimodal/text content may arrive as a list of blocks.
        # HF/Jinja chat templates used by this server expect plain strings.
        if isinstance(content, list):
            text_chunks = []
            for chunk in content:
                if isinstance(chunk, dict):
                    if chunk.get("type") == "text":
                        text_chunks.append(str(chunk.get("text", "")))
                    elif "text" in chunk:
                        text_chunks.append(str(chunk.get("text", "")))
                elif isinstance(chunk, str):
                    text_chunks.append(chunk)
            msg_copy["content"] = "".join(text_chunks)
        elif isinstance(content, dict):
            msg_copy["content"] = str(content.get("text", ""))
        elif content is None:
            msg_copy["content"] = ""
        elif not isinstance(content, str):
            msg_copy["content"] = str(content)

        tool_calls = msg_copy.get("tool_calls")
        if isinstance(tool_calls, list):
            sanitized_tool_calls = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    sanitized_tool_calls.append(call)
                    continue
                call_copy = dict(call)
                function = call_copy.get("function")
                if isinstance(function, dict):
                    function_copy = dict(function)
                    function_copy["arguments"] = _coerce_arguments_mapping(
                        function_copy.get("arguments", {})
                    )
                    call_copy["function"] = function_copy
                sanitized_tool_calls.append(call_copy)
            msg_copy["tool_calls"] = sanitized_tool_calls
        sanitized.append(msg_copy)
    return sanitized


def _sanitize_tools_for_template(tools):
    """Ensure tools payload is template-safe and has mapping parameters.

    Example transformations:
    - {"function": {"name": "f", "parameters": "not-a-dict"}}
      -> {"function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}
    - non-dict tool entries are dropped.
    - non-list input returns None.
    """
    if not isinstance(tools, list):
        return None

    sanitized = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_copy = dict(tool)
        function = tool_copy.get("function")
        if isinstance(function, dict):
            function_copy = dict(function)
            if not isinstance(function_copy.get("parameters"), dict):
                function_copy["parameters"] = {"type": "object", "properties": {}}
            tool_copy["function"] = function_copy
        sanitized.append(tool_copy)
    return sanitized


def _replace_prefix_tokens(
    eos_token_id,
    previous_turn_token_ids,
    retokeenized_previous_turn_token_ids,
    current_turn_token_ids,
):
    """Replace the token ids that are associated with the previous turn with the actual tokens
    from the previous generation (rather than the ones from the chat template application)."""

    # Strip the EOS from the previous turn token ids if it exists
    if previous_turn_token_ids and previous_turn_token_ids[-1] == eos_token_id:
        previous_turn_token_ids = previous_turn_token_ids[:-1]

    # Find the last EOS token id in the previous turn token ids
    last_eos_token_id_index = len(retokeenized_previous_turn_token_ids) - 1
    # Note that the current conversation stat may be shorter than the previous conversation state.
    scan_len = min(len(retokeenized_previous_turn_token_ids), len(current_turn_token_ids))
    for i in reversed(range(scan_len)):
        if current_turn_token_ids[i] == eos_token_id:
            last_eos_token_id_index = i
            break

    # Replace the current turn token ids with the tokens from the previous generation
    current_turn_additional_token_ids = current_turn_token_ids[last_eos_token_id_index:]

    # Return the previous turn token ids + the current turn token ids
    return previous_turn_token_ids + current_turn_additional_token_ids


def _apply_chat_template_sync(
    tokenizer, messages, tools, chat_template_kwargs, add_generation_prompt=True, _fe_rid=0
):
    """Apply the chat template and coerce to `list[int]`, for use in a worker thread.

    The coercion runs here too: it walks every token, so leaving it on the event loop
    would keep part of the stall this offload exists to remove.
    """
    # [FE-EV] This body runs ON the executor thread, so the elapsed time here is
    # tokenizer SERVICE time with no queue wait in it. Queue wait is the gap
    # between this request's tok_enq and its tok_svc. ntok/nmsg are reported
    # alongside because service time is expected to scale with them (the Jinja
    # template re-renders the whole conversation every turn).
    _t0 = time.perf_counter()
    _ids = _coerce_to_token_id_list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **chat_template_kwargs,
        )
    )
    _ev(
        "tok_svc",
        _fe_rid,
        svc_ms=f"{(time.perf_counter() - _t0) * 1000:.2f}",
        ntok=len(_ids),
        nmsg=len(messages) if messages is not None else -1,
        ntools=len(tools) if tools else 0,
        agp=int(bool(add_generation_prompt)),
    )
    return _ids


def _coerce_to_token_id_list(result):
    """Convert the return value of `tokenizer.apply_chat_template` to `list[int]`.

    transformers >= 5.x.x sometimes returns a `BatchEncoding` object instead of a `list[int]`.
    """
    # BatchEncoding / dict-like with input_ids
    if isinstance(result, dict) or hasattr(result, "input_ids"):
        ids = result["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    # Fast-tokenizer Encoding object
    if hasattr(result, "ids"):
        ids = result.ids
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return list(ids)
    # Raw tensor / ndarray
    if hasattr(result, "tolist"):
        ids = result.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return ids
    # Plain list
    return list(result)


try:
    import orjson

    HAVE_ORJSON = True
except ImportError:
    HAVE_ORJSON = False

def _logprob_null_audit(response, payload, encoder):  # pragma: no cover - diagnostic
    """[LOGPROB-AUDIT] Compare logprob arrays before and after serialization.

    Two independent logprob arrays leave this endpoint, built from two different
    engine keys, and only one of them is what NeMo-Gym consumes:

        message["generation_log_probs"]        <- result["generated_log_probs"]   (consumed)
        choice["logprobs"]["content"][j]       <- result["log_probs"]             (not consumed)

    A null in the consumed array fails downstream as
    "generation_log_probs.N: Input should be a valid number", by which point the
    response is gone and only an index survives.

    A null in the serialized output is ambiguous on its own: orjson emits NaN,
    Infinity and -Infinity as null (per its docs), so a null can mean the engine
    produced a non-finite float OR a literal None. This audit resolves that by
    reporting the *pre-serialization* Python value -- its repr and its type --
    next to the post-serialization null. repr 'nan'/'-inf' means the engine
    produced a non-finite float and the encoder converted it; repr 'None' means
    the engine emitted None and the encoder is not implicated.

    Fires only when something is actually wrong, so it costs one walk of the
    arrays per request and prints nothing on the happy path.
    """
    try:
        def scan(seq):
            bad = []
            if not isinstance(seq, list):
                return bad
            for i, v in enumerate(seq):
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    bad.append(i)
            return bad

        # Pre-serialization view of the Python objects.
        pre = []
        for ci, ch in enumerate(response.get("choices") or []):
            msg = ch.get("message") or {}
            pre.append(
                {
                    "choice": ci,
                    "gen": scan(msg.get("generation_log_probs")),
                    "gen_len": len(msg.get("generation_log_probs") or []),
                    "content": scan(
                        [
                            e.get("logprob")
                            for e in ((ch.get("logprobs") or {}).get("content") or [])
                        ]
                    ),
                    "finish_reason": ch.get("finish_reason"),
                    "n_gen_tokens": len(msg.get("generation_token_ids") or []),
                }
            )

        # Post-serialization view: re-parse exactly what goes on the wire.
        post = []
        if payload is not None:
            parsed = json.loads(
                payload.decode() if isinstance(payload, (bytes, bytearray)) else payload
            )
            for ci, ch in enumerate(parsed.get("choices") or []):
                msg = ch.get("message") or {}
                post.append(
                    {
                        "choice": ci,
                        "gen": scan(msg.get("generation_log_probs")),
                        "content": scan(
                            [
                                e.get("logprob")
                                for e in ((ch.get("logprobs") or {}).get("content") or [])
                            ]
                        ),
                    }
                )

        interesting = any(p["gen"] or p["content"] for p in pre) or any(
            p["gen"] or p["content"] for p in post
        )
        if not interesting:
            return

        print(
            f"[LOGPROB-AUDIT] pid={os.getpid()} encoder={encoder} choices={len(pre)}",
            flush=True,
        )
        for p in pre:
            ci = p["choice"]
            q = next((x for x in post if x["choice"] == ci), {"gen": [], "content": []})
            print(
                f"[LOGPROB-AUDIT]   choice={ci} finish_reason={p['finish_reason']!r} "
                f"n_gen_tokens={p['n_gen_tokens']} gen_len={p['gen_len']} "
                f"pre.generation_log_probs={p['gen'][:10]} "
                f"post.generation_log_probs={q['gen'][:10]} "
                f"pre.logprobs.content={p['content'][:10]} "
                f"post.logprobs.content={q['content'][:10]}",
                flush=True,
            )
            # The decisive detail: what the value was BEFORE the encoder saw it.
            msg = (response.get("choices") or [])[ci].get("message") or {}
            seq = msg.get("generation_log_probs") or []
            for i in (p["gen"] or q["gen"])[:5]:
                if i < len(seq):
                    v = seq[i]
                    gt = msg.get("generation_token_ids") or []
                    print(
                        f"[LOGPROB-AUDIT]     idx={i} pre_value={v!r} pre_type={type(v).__name__} "
                        f"token_id={gt[i] if i < len(gt) else None} "
                        f"neighbors={[repr(x) for x in seq[max(0, i - 2):i + 3]]}",
                        flush=True,
                    )
    except Exception as exc:  # noqa: BLE001 - diagnostic must never break a response
        print(f"[LOGPROB-AUDIT] audit failed: {type(exc).__name__}: {exc}", flush=True)


# [ORJSON] Which encoder serializes the chat response, printed once per replica.
# This decides how a non-finite logprob leaves the process: orjson emits
# "Nan, Infinity, and -Infinity ... as null" (its docs), which arrives at Gym as
# None and fails pydantic as "generation_log_probs.N: Input should be a valid
# number". stdlib jsonify instead emits a bare NaN literal, which json.loads
# accepts as float('nan') and pydantic accepts as a valid float. So the null-vs-NaN
# question upstream is decided entirely by this flag.
print(f"[ORJSON] pid={os.getpid()} HAVE_ORJSON={HAVE_ORJSON}", flush=True)



try:
    from quart import Blueprint, Response, current_app, g, jsonify, request

    bp = Blueprint('chat_completions_api', __name__)

    @bp.before_request
    async def _fe_before_request():
        g._fe_rid = next(_fe_rid)
        g._fe_t0 = time.perf_counter()
        _fe_gauge["handler"] += 1
        _ev("recv", g._fe_rid)

    @bp.teardown_request
    async def _fe_teardown_request(exc):
        # teardown runs on every exit path -- success, early 4xx/5xx return and
        # exception -- so the handler gauge cannot leak.
        rid = getattr(g, "_fe_rid", None)
        if rid is None:
            return
        _fe_gauge["handler"] -= 1
        _ev(
            "done",
            rid,
            total_ms=f"{(time.perf_counter() - g._fe_t0) * 1000:.1f}",
            err=1 if exc is not None else 0,
        )

    def apply_parsers(
        message_text, tools, parsers_list, tools_requested, chat_template_kwargs=None
    ):
        """Runs CPU-intensive text parsing."""
        meta = {}
        for parser in parsers_list:
            if parser not in PARSER_MAPPING:
                raise ValueError(f"Parser {parser} not found in PARSER_MAPPING")

            prev_text = message_text
            parsed_text, new_info = PARSER_MAPPING[parser].parse(
                message_text, tools=tools, chat_template_kwargs=chat_template_kwargs
            )
            if "tool_calls" in new_info:
                new_info["tool_calls"] = _normalize_tool_calls(
                    new_info.get("tool_calls", []), tools=tools
                )
                if not tools_requested:
                    # Ignore incidental tool-call syntax in plain chat mode.
                    parsed_text = prev_text
                    new_info.pop("tool_calls", None)
            message_text = parsed_text

            assert not (
                meta.keys() & new_info.keys()
            ), "Multiple parsers found the same information."
            meta.update(new_info)

        return message_text, meta

    @bp.route('/chat/completions', methods=['POST'])
    @bp.route('/v1/chat/completions', methods=['POST'])
    async def chat_completions():
        """Handles async POST requests for chat completions."""
        client = current_app.config['client']
        tokenizer = current_app.config['tokenizer']
        parsers = current_app.config['parsers']
        block_size_tokens = current_app.config.get('block_size_tokens')
        coordinator_policy = current_app.config.get('prefix_caching_coordinator_policy')

        req = await request.get_json()

        # [FE-REQ] see _FE_DUMP_N above. Emitted before any parsing so a request
        # that later 400s or throws still shows up.
        try:
            _rid_req = getattr(g, "_fe_rid", 0)
            _msgs = req.get("messages") or []
            if _FE_REQ_SUMMARY:
                print(
                    f"[FE-REQ] pid={os.getpid()} rid={_rid_req}"
                    f" keys={sorted(req.keys())}"
                    f" nmsg={len(_msgs)}"
                    f" roles={[m.get('role') for m in _msgs][:16]}"
                    f" ntools={len(req.get('tools') or [])}"
                    f" tool_choice={req.get('tool_choice')!r}"
                    f" parallel_tool_calls={req.get('parallel_tool_calls')!r}"
                    f" chat_template_kwargs={req.get('chat_template_kwargs')!r}"
                    f" temperature={req.get('temperature')!r}"
                    f" top_p={req.get('top_p')!r} top_k={req.get('top_k')!r}"
                    f" seed={req.get('seed')!r}"
                    f" max_tokens={req.get('max_tokens')!r}"
                    f" max_completion_tokens={req.get('max_completion_tokens')!r}"
                    f" stop={req.get('stop')!r} stream={req.get('stream')!r}"
                    f" logprobs={req.get('logprobs')!r}"
                    f" top_logprobs={req.get('top_logprobs')!r}"
                    f" chars={sum(len(str(m.get('content') or '')) for m in _msgs)}",
                    flush=True,
                )
        except Exception as _e:
            print(f"[FE-REQ] dump failed: {type(_e).__name__}: {_e}", flush=True)

        tools = req.get("tools", None)
        tool_choice = req.get("tool_choice", None)
        parallel_tool_calls = req.get("parallel_tool_calls", True)
        tools_requested = bool(tools) and tool_choice != "none"
        messages = req.get("messages")
        chat_template_kwargs = req.get("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, dict):
            logger.warning(
                "Ignoring non-dict chat_template_kwargs: %s", type(chat_template_kwargs).__name__
            )
            chat_template_kwargs = {}
        # --- 1. Parse Messages ---
        if not messages:
            return Response("Missing 'messages' field", status=400)
        if not isinstance(messages, list):
            return Response("'messages' must be a list", status=400)
        template_messages = _sanitize_messages_for_template(messages)
        template_tools = _sanitize_tools_for_template(tools)

        try:
            if (
                hasattr(tokenizer, 'apply_chat_template')
                and getattr(tokenizer, "chat_template", None) is not None
            ):
                # [FE-EV] tok_enq -> tok_deq spans queue wait + service; the
                # tok_svc event emitted inside the executor isolates service.
                _fe_rid_cur = getattr(g, "_fe_rid", 0)
                _fe_gauge["tokenize"] += 1
                _ev("tok_enq", _fe_rid_cur)
                _fe_tok_t0 = time.perf_counter()
                try:
                    prompt_tokens = await asyncio.get_running_loop().run_in_executor(
                        current_app.config['tokenize_executor'],
                        partial(
                            _apply_chat_template_sync,
                            current_app.config['tokenize_tokenizer'],
                            template_messages,
                            template_tools,
                            chat_template_kwargs,
                            _fe_rid=_fe_rid_cur,
                        ),
                    )
                finally:
                    _fe_gauge["tokenize"] -= 1
                    _ev(
                        "tok_deq",
                        _fe_rid_cur,
                        wait_ms=f"{(time.perf_counter() - _fe_tok_t0) * 1000:.1f}",
                    )

                if req.get("prevent_retokenization", True):
                    # If we are avoiding retokenization, we need to replace some prompt tokens with the prompt/generation tokens from the previous generation
                    # This improves prefix cache hits and reduces logprob variation between training and inference.

                    # Find the last assistant message
                    last_assistant_message_idx = None
                    for i in reversed(range(len(template_messages))):
                        if template_messages[i]["role"] == "assistant":
                            last_assistant_message_idx = i
                            break

                    last_assistant_message = (
                        template_messages[last_assistant_message_idx]
                        if last_assistant_message_idx is not None
                        else None
                    )

                    # Only proceed if the last assistant message has the token IDs from a previous generation.
                    # Dataset-provided conversation history won't have these fields.
                    if (
                        last_assistant_message is not None
                        and "prompt_token_ids" in last_assistant_message
                        and "generation_token_ids" in last_assistant_message
                    ):
                        eos_token_id = tokenizer.eos_id
                        assert eos_token_id is not None, "Your tokenizer must have an EOS token ID!"

                        warnings.warn(
                            "Avoiding prefix retokenization."
                            " This is a patch that ensures subsequent generations are not retokenized differently than the previous generation."
                            " This may cause unexpected behavior if messages (including system messages) are altered between generations."
                        )

                        messages_to_last_assistant_message = template_messages[
                            : last_assistant_message_idx + 1
                        ]

                        # Get the templated tokenization of just the previous generation
                        # [FE-EV] second trip through the same single-worker
                        # executor, so it queues behind the same thread.
                        _fe_rid2 = getattr(g, "_fe_rid", 0)
                        _fe_gauge["tokenize"] += 1
                        _ev("tok_enq", _fe_rid2, pass_=2)
                        _fe_tok2_t0 = time.perf_counter()
                        try:
                            retokenized_previous_turn_token_ids = (
                                await asyncio.get_running_loop().run_in_executor(
                                    current_app.config['tokenize_executor'],
                                    partial(
                                        _apply_chat_template_sync,
                                        current_app.config['tokenize_tokenizer'],
                                        messages_to_last_assistant_message,
                                        template_tools,
                                        chat_template_kwargs,
                                        add_generation_prompt=False,
                                        _fe_rid=_fe_rid2,
                                    ),
                                )
                            )
                        finally:
                            _fe_gauge["tokenize"] -= 1
                            _ev(
                                "tok_deq",
                                _fe_rid2,
                                pass_=2,
                                wait_ms=f"{(time.perf_counter() - _fe_tok2_t0) * 1000:.1f}",
                            )

                        # Replace the prefix tokens with the tokens from the previous generation.
                        # If prior token IDs are unavailable, fall back to normal retokenized prompt
                        # instead of failing the request.
                        prompt_token_ids = last_assistant_message.get("prompt_token_ids")
                        generation_token_ids = last_assistant_message.get("generation_token_ids")

                        if isinstance(prompt_token_ids, list) and isinstance(
                            generation_token_ids, list
                        ):
                            previous_turn_token_ids = prompt_token_ids + generation_token_ids
                            prompt_tokens = _replace_prefix_tokens(
                                eos_token_id,
                                previous_turn_token_ids,
                                retokenized_previous_turn_token_ids,
                                prompt_tokens,
                            )
                        else:
                            logger.warning(
                                "Last assistant message missing prompt_token_ids/"
                                "generation_token_ids; skipping prefix replacement."
                            )

            else:
                warnings.warn(
                    "Tokenizer does not support 'apply_chat_template'. Using tokenize instead."
                )
                prompt_tokens = tokenizer.tokenize(
                    "\n".join([message["content"] for message in messages])
                )
        except Exception as e:
            logger.error(f"{traceback.format_exc()}")
            return Response(f"Error processing 'messages': {e}", status=500)

        # --- 2. Parse Sampling Params ---
        try:
            temperature = float(_get_non_none(req, "temperature", 1.0))
            top_p = float(_get_non_none(req, "top_p", 1.0))
            top_k = int(_get_non_none(req, "top_k", 0))
            n = int(_get_non_none(req, "n", 1))  # Number of choices to generate

            if temperature == 0.0:
                top_k = 1
                top_p = 0.0

            # Check for 'logprobs' (bool) and 'top_logprobs' (int)
            return_log_probs = bool(_get_non_none(req, "logprobs", False))
            top_n_logprobs = int(_get_non_none(req, "top_logprobs", 0)) if return_log_probs else 0
            skip_prompt_log_probs = bool(_get_non_none(req, "skip_prompt_log_probs", True))
            add_BOS = bool(_get_non_none(req, "add_BOS", False))

            # The engine only handles add_BOS for string prompts, not pre-tokenized
            # input. Since we pre-tokenize via apply_chat_template, we must handle
            # BOS ourselves, matching the logic in tokenize_prompt().
            if hasattr(tokenizer, 'bos') and tokenizer.bos is not None:
                start_idx = 0
                while start_idx < len(prompt_tokens) and prompt_tokens[start_idx] == tokenizer.bos:
                    start_idx += 1
                if start_idx > 0:
                    prompt_tokens = prompt_tokens[start_idx:]

                if add_BOS:
                    prompt_tokens = [tokenizer.bos] + prompt_tokens

            max_tokens = req.get("max_completion_tokens", None) or req.get("max_tokens", None)
            ignore_eos = bool(req.get("ignore_eos", False))

            # Does the client want the prompt tokens echoed back? Only then does the
            # engine need to keep the prompt_tokens tensor on the response payload.
            # return_tokenized_data (implied by prevent_retokenization) needs the ids;
            # return_raw_text needs the ids to detokenize the prompt into raw_text.
            prevent_retokenization = req.get("prevent_retokenization", True)
            return_tokenized_data = (
                req.get("return_tokenized_data", False) or prevent_retokenization
            )
            return_raw_text = req.get("return_raw_text", False)
            return_prompt_tokens = return_tokenized_data or return_raw_text

            sampling_params = SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_log_probs=return_log_probs,
                top_n_logprobs=top_n_logprobs,
                num_tokens_to_generate=(int(max_tokens) if max_tokens is not None else None),
                skip_prompt_log_probs=skip_prompt_log_probs,
                add_BOS=add_BOS,
                termination_id=-1 if ignore_eos else None,
                return_prompt_tokens=return_prompt_tokens,
                streaming_interval=int(_get_non_none(req, "streaming_interval", 1)),
                # This frontend detokenizes its own output below. Keeping it off the
                # coordinator matters because that is one process shared by all DP ranks.
                detokenize_generations=False,
            )
        except ValueError as e:
            return Response(f"Invalid sampling parameter: {e}", status=400)

        # --- 3. Send Requests to Engine ---
        # Hash here rather than at the coordinator: the tokens are already in hand,
        # frontends run many-to-one against a single serial coordinator loop, and
        # hashing there would mean unpacking the prompt frame the split was
        # introduced to avoid. Skipped unless the coordinator routes on prefix
        # affinity, since otherwise nobody reads it and the frame is never sent.
        block_hashes = (
            compute_block_hashes_batched(
                torch.tensor(prompt_tokens, dtype=torch.int64), block_size_tokens
            )
            if block_size_tokens and routes_on_prefix(coordinator_policy)
            else None
        )

        stream_requested = bool(req.get("stream", False))
        if stream_requested:
            # Streaming currently supports only Hugging Face fast tokenizers.
            try:
                incremental_detokenizers = [
                    HuggingFaceFastIncrementalDetokenizer(tokenizer, prompt_tokens)
                    for _ in range(n)
                ]
            except ValueError as error:
                return Response(str(error), status=400)

            streams = [
                client.add_request_streaming(prompt_tokens, sampling_params, block_hashes)
                for _ in range(n)
            ]
            include_usage = bool((req.get("stream_options") or {}).get("include_usage", False))
            response = Response(
                openai_stream(
                    streams,
                    tokenizer,
                    incremental_detokenizers,
                    chat=True,
                    return_log_probs=return_log_probs,
                    include_usage=include_usage,
                ),
                content_type="text/event-stream",
            )
            response.timeout = None
            return response

        tasks = [
            client.add_request(prompt_tokens, sampling_params, block_hashes) for _ in range(n)
        ]

        if current_app.config['verbose']:
            start_time = time.perf_counter()

        # [FE-EV] generate stage: submitted -> all n engine replies back.
        # n is logged so HTTP-level and engine-level counts convert cleanly
        # (one HTTP request fans out to n engine requests).
        _fe_rid_gen = getattr(g, "_fe_rid", 0)
        _fe_gauge["generate"] += 1
        _ev("gen_beg", _fe_rid_gen, n=len(tasks), ntok=len(prompt_tokens))
        _fe_gen_t0 = time.perf_counter()
        try:
            batch_results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return Response(f"Error during inference: {e}", status=500)
        finally:
            _fe_gauge["generate"] -= 1
            _ev(
                "gen_end",
                _fe_rid_gen,
                gen_ms=f"{(time.perf_counter() - _fe_gen_t0) * 1000:.1f}",
            )

        if current_app.config['verbose']:
            logging.info(
                f"Batch of {len(tasks)} requests (n={n}) processed in "
                f"{time.perf_counter() - start_time:.2f}s"
            )

        # --- 4. Check for failed requests ---
        failed_errors = []
        has_nontransient_error = False
        for i, record in enumerate(batch_results):
            if record.get("status") == "FAILED":
                events = record.get("events", [])
                error_events = [
                    e for e in events if e.get("type") in ("ERROR_NONTRANSIENT", "ERROR_TRANSIENT")
                ]
                if any(e.get("type") == "ERROR_NONTRANSIENT" for e in error_events):
                    has_nontransient_error = True
                error_msg = (
                    str(error_events[-1].get("payload", "Unknown error"))
                    if error_events
                    else "Unknown error"
                )
                failed_errors.append(f"Request {i}: {error_msg}")

        if failed_errors:
            error_detail = "; ".join(failed_errors)
            status = 400 if has_nontransient_error else 500
            logger.error(f"Inference request(s) failed: {error_detail}")

            # NOTE: This exact string is required for compatibility with Nemo-RL, DO NOT MODIFY.
            if "MaxSequenceLengthOverflowError" in error_detail:
                error_msg = (
                    f"This model's maximum context length was exceeded. "
                    f"Your messages resulted in {len(prompt_tokens)} tokens. "
                    f"Please reduce the length of the messages. {error_detail}"
                )
                return Response(error_msg, status=400)

            return Response(f"Inference request(s) failed: {error_detail}", status=status)

        # --- 5. Format OpenAI Response ---
        choices = []
        total_completion_tokens = 0
        prompt_tokens_counts = []
        cached_tokens_counts = []

        # return_tokenized_data / return_raw_text / return_prompt_tokens were computed
        # at submit time (above) and drive both the response shape here and whether the
        # engine kept the prompt_tokens tensor on the payload.
        request_idx = 0
        for result_item in batch_results:
            result = unwrap_serialized_tensors(result_item)

            text_output = TextGenerationController.detokenize(
                tokenizer,
                result["generated_tokens"],
                remove_EOD=not sampling_params.detokenize_stop_sequence,
            )
            # The engine always reports prompt_length (for usage), but drops the
            # prompt_tokens tensor unless return_prompt_tokens was set.
            prompt_tokens_count = result.get("prompt_length")
            if prompt_tokens_count is None:
                prompt_tokens_out = result["prompt_tokens"]
                prompt_tokens_count = len(prompt_tokens_out) if prompt_tokens_out is not None else 0
            prompt_tokens_counts.append(prompt_tokens_count)
            cached_tokens_counts.append(result.get("num_cached_tokens", 0))

            logprobs_content = None
            if sampling_params.return_log_probs:
                token_logprobs = result.get('log_probs', [])

                tokens_to_decode = [[tok] for tok in result["generated_tokens"]]
                tokens = list(map(tokenizer.detokenize, tokens_to_decode))

                # Get top_n_logprobs if available
                generated_top_n_logprobs = result.get('generated_top_n_logprobs')

                logprobs_content = []
                for i, (tok, lp) in enumerate(zip(tokens, token_logprobs)):
                    # Build top_logprobs list for this token position
                    top_logprobs_list = []
                    if generated_top_n_logprobs and i < len(generated_top_n_logprobs):
                        top_n_dict = generated_top_n_logprobs[i]
                        for token_str, logprob in top_n_dict.items():
                            top_logprobs_list.append(
                                {
                                    "token": token_str,
                                    "logprob": logprob,
                                    "bytes": list(token_str.encode("utf-8")),
                                }
                            )

                    logprobs_content.append(
                        {
                            "token": tok,
                            "logprob": lp,
                            "bytes": list(tok.encode("utf-8")),
                            "top_logprobs": top_logprobs_list,
                        }
                    )

                # [FE-LOGPROB] Non-finite logprobs at the point of creation.
                # Test for NaN/inf, NOT None: `lp` here is still a Python float, and
                # the response is serialized with orjson (see the Response(...) below),
                # which per its docs emits "Nan, Infinity, and -Infinity ... as null".
                # So a NaN produced by the engine leaves this process as JSON null,
                # arrives at Gym as None, passes untouched through the
                # return_tokenized_data path, and finally fails as a pydantic error
                # "generation_log_probs.N: Input should be a valid number" -- by which
                # point the request is gone and only an index survives. That 500 then
                # becomes stream_error in the collector and strands every prompt group
                # not yet yielded, forcing a wholesale batch re-dispatch. Checking for
                # None alone would never fire here. Diagnostic only.
                _bad_idx = [
                    i
                    for i, e in enumerate(logprobs_content)
                    if e["logprob"] is None
                    or (isinstance(e["logprob"], float) and not math.isfinite(e["logprob"]))
                ]
                if _bad_idx:
                    print(
                        f"[FE-LOGPROB] pid={os.getpid()} bad={len(_bad_idx)}/"
                        f"{len(logprobs_content)} idx={_bad_idx[:10]} "
                        f"vals={[logprobs_content[i]['logprob'] for i in _bad_idx[:5]]!r} "
                        f"n_tokens={len(tokens)} n_log_probs={len(token_logprobs)} "
                        f"n_gen_tokens={len(result.get('generated_tokens') or [])} "
                        f"finish_reason={result.get('finish_reason')!r} "
                        f"cached_tokens={result.get('num_cached_tokens', 0)}",
                        flush=True,
                    )

            metadata = {}
            message_text = text_output

            if parsers:
                message_text, metadata = apply_parsers(
                    message_text,
                    tools,
                    parsers,
                    tools_requested,
                    chat_template_kwargs=chat_template_kwargs,
                )

            normalized_tool_calls = metadata.get("tool_calls", [])

            # Apply parallel_tool_calls filtering (matches vLLM behavior)
            normalized_tool_calls = _maybe_filter_parallel_tool_calls(
                normalized_tool_calls, parallel_tool_calls
            )

            # Determine content based on tool_choice (matches vLLM behavior):
            # - Named tool choice or "required": content is empty string
            # - Otherwise: content is the parsed message text
            is_named_tool_choice = isinstance(tool_choice, dict) and "function" in tool_choice
            if normalized_tool_calls and (is_named_tool_choice or tool_choice == "required"):
                content = ""
            else:
                content = message_text if message_text is not None else ""

            message = {"role": "assistant", "content": content}
            if normalized_tool_calls:
                message["tool_calls"] = normalized_tool_calls
            if "reasoning" in metadata:
                message["reasoning_content"] = metadata["reasoning"]

            if return_tokenized_data:
                message["prompt_token_ids"] = result["prompt_tokens"]
                message["generation_token_ids"] = result["generated_tokens"]
            if return_raw_text:
                prompt_str = tokenizer.detokenize(result["prompt_tokens"])
                message["raw_text"] = prompt_str + text_output
            # Small RL/debug scalars (a few bytes each); harmless to keep for
            # NeMo-RL compatibility.
            message["generation_log_probs"] = result.get("generated_log_probs", [])
            message["policy_epoch"] = result["policy_epoch"]
            message["kv_cache_epoch"] = result["kv_cache_epoch"]
            message["num_evictions"] = sum(1 for e in result["events"] if e.get("type") == "EVICT")
            return_log_probs = sampling_params.return_log_probs

            # Determine finish_reason following vLLM conventions:
            # - "tool_calls" for auto or required tool choice when tools are called
            # - "stop" for named tool choice (even when tools are called)
            # - "length" when max tokens is reached
            if (
                len(result["generated_tokens"])
                >= result["sampling_params"]["num_tokens_to_generate"]
            ):
                finish_reason = "length"
            elif normalized_tool_calls and not is_named_tool_choice:
                finish_reason = "tool_calls"
            else:
                finish_reason = "stop"

            # Choice-level prompt/generation_token_ids, generation_log_probs and
            # raw_text were duplicates of message-level data (or reconstructable);
            # dropped to match vLLM's response shape and cut payload size.
            choice_data = {
                "index": request_idx,
                "message": message,
                # 'logprobs' in chat API is an object containing 'content'
                "logprobs": {"content": logprobs_content} if return_log_probs else None,
                "finish_reason": finish_reason,
            }
            if current_app.config['verbose']:
                logging.info(_redact_token_id_lists_for_logging(result))

            if result["routing_indices"] is not None:
                choice_data["moe_topk_indices"] = result["routing_indices"]
                if prompt_tokens_count:
                    choice_data["prompt_moe_topk_indices"] = result["routing_indices"][
                        :prompt_tokens_count
                    ]

            choices.append(choice_data)
            if result.get("generated_log_probs") is None:
                logger.warning(
                    "Generation log probs is None for request:\n%s",
                    json.dumps(_redact_token_id_lists_for_logging(result), indent=4),
                )
            total_completion_tokens += len(result["generated_tokens"])
            request_idx += 1

        prompt_token_count = max(prompt_tokens_counts) if prompt_tokens_counts else 0
        cached_token_count = max(cached_tokens_counts) if cached_tokens_counts else 0
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "created": int(time.time()),
            "model": "EMPTY",
            "object": "chat.completion",
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": total_completion_tokens,
                "total_tokens": prompt_token_count + total_completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_token_count},
            },
        }

        if HAVE_ORJSON:
            # Use orjson for faster serialization
            _payload = orjson.dumps(response)
            # [LOGPROB-AUDIT] Audit the exact bytes going on the wire against the
            # Python objects that produced them. Only prints when a null or
            # non-finite value is present on either side.
            _logprob_null_audit(response, _payload, "orjson")
            return Response(_payload, mimetype="application/json")
        else:
            _logprob_null_audit(response, None, "jsonify")
            return jsonify(response)

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
