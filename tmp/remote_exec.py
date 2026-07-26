from __future__ import annotations

import argparse
import os
import sys

import paramiko


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()

    password = os.environ.get("SVG_REMOTE_PASSWORD")
    if not password:
        raise SystemExit("SVG_REMOTE_PASSWORD is required")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        username=args.user,
        password=password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    _, stdout, stderr = client.exec_command(args.command)
    stdout.channel.settimeout(None)
    out = stdout.read()
    err = stderr.read()
    status = stdout.channel.recv_exit_status()
    client.close()

    sys.stdout.buffer.write(out)
    sys.stderr.buffer.write(err)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
