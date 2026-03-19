"""
Snowflake Connection Utility

Provides a singleton Snowflake Snowpark session with two credential sources:
1. st.secrets (Streamlit Cloud) — tried first
2. .env file (local development) — fallback

Returns None gracefully if no credentials are available.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from snowflake.snowpark import Session
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization


_session = None
_session_attempted = False


def _get_secrets_credentials() -> Optional[dict]:
    """Try to load credentials from st.secrets (Streamlit Cloud)."""
    try:
        import streamlit as st
        secrets = st.secrets
        account = secrets.get("SNOWFLAKE_ACCOUNT")
        if not account:
            return None
        params = {
            "account": account,
            "user": secrets["SNOWFLAKE_USER"],
            "role": secrets.get("SNOWFLAKE_ROLE", "TRAINING_ROLE"),
            "warehouse": secrets.get("SNOWFLAKE_WAREHOUSE", "COPILOT_WH"),
            "database": secrets.get("SNOWFLAKE_DATABASE", "ANALYTICS_COPILOT"),
        }
        private_key_str = secrets.get("SNOWFLAKE_PRIVATE_KEY")
        if private_key_str:
            private_key = serialization.load_pem_private_key(
                private_key_str.encode(),
                password=None,
                backend=default_backend(),
            )
            pkb = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            params["private_key"] = pkb
        elif secrets.get("SNOWFLAKE_PASSWORD"):
            params["password"] = secrets["SNOWFLAKE_PASSWORD"]
        else:
            return None
        return params
    except (AttributeError, FileNotFoundError, KeyError, Exception):
        return None


def _get_env_credentials() -> Optional[dict]:
    """Load credentials from .env file (local development)."""
    load_dotenv()
    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER"]
    if any(not os.getenv(v) for v in required):
        return None
    params = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "role": os.getenv("SNOWFLAKE_ROLE", "TRAINING_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COPILOT_WH"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "ANALYTICS_COPILOT"),
    }
    private_key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    if private_key_path:
        key_path = Path(private_key_path).expanduser()
        if not key_path.exists():
            return None
        with open(key_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend(),
            )
        pkb = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        params["private_key"] = pkb
    elif password:
        params["password"] = password
    else:
        return None
    return params


def get_session() -> Optional[Session]:
    """
    Get or create a Snowflake Snowpark session.

    Tries st.secrets first (Streamlit Cloud), then .env (local dev).
    Returns None if no credentials are available (graceful fallback).
    Caches the result (including None) to avoid repeated attempts.
    """
    global _session, _session_attempted

    if _session_attempted:
        return _session

    _session_attempted = True

    params = _get_secrets_credentials() or _get_env_credentials()
    if params is None:
        print("No Snowflake credentials found. Running in disconnected mode.")
        return None

    try:
        _session = Session.builder.configs(params).create()
        print(f"Snowflake session created for user: {params['user']}")
        return _session
    except Exception as e:
        print(f"Failed to create Snowflake session: {e}")
        return None


def close_session() -> None:
    """Close the active Snowflake session and reset the singleton."""
    global _session, _session_attempted
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
    _session = None
    _session_attempted = False


def reset_session() -> None:
    """Reset the singleton so the next get_session() call retries."""
    global _session, _session_attempted
    _session = None
    _session_attempted = False
