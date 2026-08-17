from __future__ import annotations

import argparse
import os

import paramiko


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--local", required=True)
    parser.add_argument("--remote", required=True)
    args = parser.parse_args()

    password = os.environ.get("SVG_REMOTE_PASSWORD")
    if not password:
        raise SystemExit("SVG_REMOTE_PASSWORD is required")

    transport = paramiko.Transport((args.host, 22))
    transport.connect(username=args.user, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        sftp.put(args.local, args.remote)
    finally:
        sftp.close()
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
