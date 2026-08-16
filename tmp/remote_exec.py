from __future__ import annotations

import argparse
import base64
import os
import sys

import paramiko


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument("--command")
    command_group.add_argument("--command-b64")
    command_group.add_argument(
        "--download",
        nargs=2,
        action="append",
        metavar=("REMOTE_PATH", "LOCAL_PATH"),
    )
    command_group.add_argument(
        "--upload",
        nargs=2,
        action="append",
        metavar=("LOCAL_PATH", "REMOTE_PATH"),
    )
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
    if args.download:
        with client.open_sftp() as sftp:
            for remote_path, local_path in args.download:
                sftp.get(remote_path, local_path)
        client.close()
        return 0
    if args.upload:
        with client.open_sftp() as sftp:
            for local_path, remote_path in args.upload:
                sftp.put(local_path, remote_path)
        client.close()
        return 0

    command = (
        base64.b64decode(args.command_b64).decode("utf-8")
        if args.command_b64
        else args.command
    )
    _, stdout, stderr = client.exec_command(command)
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
