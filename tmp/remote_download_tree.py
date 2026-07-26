from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

import paramiko


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--local", required=True)
    args = parser.parse_args()

    password = os.environ.get("SVG_REMOTE_PASSWORD")
    if not password:
        raise SystemExit("SVG_REMOTE_PASSWORD is required")

    transport = paramiko.Transport((args.host, 22))
    transport.connect(username=args.user, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    def download(remote: str, local: Path) -> None:
        metadata = sftp.stat(remote)
        if stat.S_ISDIR(metadata.st_mode):
            local.mkdir(parents=True, exist_ok=True)
            for item in sftp.listdir_attr(remote):
                download(f"{remote.rstrip('/')}/{item.filename}", local / item.filename)
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote, str(local))

    try:
        download(args.remote, Path(args.local))
    finally:
        sftp.close()
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
