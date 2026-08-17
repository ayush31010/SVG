from __future__ import annotations

import argparse
import os
import re
import sys
import time

import paramiko


def _read_available(channel: paramiko.Channel) -> str:
    chunks: list[bytes] = []
    while channel.recv_ready():
        chunks.append(channel.recv(4096))
    return b"".join(chunks).decode(errors="replace")


def _wait_for(channel: paramiko.Channel, pattern: str, timeout: float = 20.0) -> str:
    transcript = ""
    deadline = time.monotonic() + timeout
    compiled = re.compile(pattern, re.IGNORECASE)
    while time.monotonic() < deadline:
        transcript += _read_available(channel)
        if compiled.search(transcript):
            return transcript
        if channel.exit_status_ready():
            transcript += _read_available(channel)
            break
        time.sleep(0.05)
    raise RuntimeError(f"timed out waiting for passwd prompt matching {pattern!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rotate a password over SSH without logging it")
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    args = parser.parse_args()

    old_password = os.environ.get("SVG_REMOTE_PASSWORD")
    new_password = os.environ.get("SVG_REMOTE_NEW_PASSWORD")
    if not old_password or not new_password:
        raise SystemExit("SVG_REMOTE_PASSWORD and SVG_REMOTE_NEW_PASSWORD are required")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        username=args.user,
        password=old_password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    channel = client.invoke_shell(width=120, height=24)
    try:
        _wait_for(channel, r"[$#>]\s*$")
        channel.send("passwd\n")
        first = _wait_for(
            channel,
            r"(?:current.*password|new.*password|password:)\s*$",
        )
        if re.search(r"current.*password|\(current\).*password", first, re.IGNORECASE):
            channel.send(old_password + "\n")
            _wait_for(channel, r"new.*password\s*:\s*$")
        channel.send(new_password + "\n")
        _wait_for(channel, r"(?:retype|repeat|again).*password\s*:\s*$")
        channel.send(new_password + "\n")
        transcript = _wait_for(
            channel,
            r"(?:password.*(?:updated|changed|success)|[$#>]\s*$)",
            timeout=30.0,
        )
        if re.search(r"failure|unchanged|authentication token manipulation|too simple|bad password", transcript, re.IGNORECASE):
            raise RuntimeError("remote passwd rejected the new password")
    finally:
        channel.close()
        client.close()

    print("Remote password changed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Password rotation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
