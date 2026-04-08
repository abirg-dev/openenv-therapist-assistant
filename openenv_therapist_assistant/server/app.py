# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Openenv Therapist Assistant Environment.

This module creates an HTTP server that exposes the OpenenvTherapistAssistantEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn openenv_therapist_assistant.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn openenv_therapist_assistant.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m openenv_therapist_assistant.server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import (
        OpenenvTherapistAssistantActionModel,
        OpenenvTherapistAssistantObservation,
    )
    from .openenv_therapist_assistant_environment import OpenenvTherapistAssistantEnvironment
except ModuleNotFoundError:
    from models import OpenenvTherapistAssistantActionModel, OpenenvTherapistAssistantObservation
    from server.openenv_therapist_assistant_environment import OpenenvTherapistAssistantEnvironment


# Create the app with web interface and README integration
app = create_app(
    OpenenvTherapistAssistantEnvironment,
    OpenenvTherapistAssistantActionModel,
    OpenenvTherapistAssistantObservation,
    env_name="openenv_therapist_assistant",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m openenv_therapist_assistant.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn openenv_therapist_assistant.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
