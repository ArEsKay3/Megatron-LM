# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optional extension point for formatting completed chat responses."""

from typing import Any, Protocol


class ChatResponseFormatter(Protocol):
    """Picklable formatter passed to every HTTP frontend replica.

    Implementations validate requests before inference and format completed
    engine replies. They own parsing as well as the OpenAI response schema.
    """

    def validate_request(self, body: dict[str, Any]) -> None:
        """Raise ValueError for unsupported requests, before submitting work."""
        ...

    async def format_response(
        self,
        body: dict[str, Any],
        prompt_tokens: list[int],
        results: list[dict[str, Any]],
        texts: list[str],
    ) -> dict[str, Any]:
        """Format deserialized, successfully completed inference replies."""
        ...
