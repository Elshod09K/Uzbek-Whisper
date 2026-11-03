"""Admin CLI.

    python -m webapp.server.manage add-user NAME --role admin|reviewer   (asks for password)
    python -m webapp.server.manage worker-token
"""
import argparse
import getpass
import time

from . import db


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add-user")
    a.add_argument("username")
    a.add_argument("--role", choices=["admin", "reviewer"], default="reviewer")
    a.add_argument("--password", help="omit to be asked interactively")
    sub.add_parser("worker-token")
    args = p.parse_args()
    db.init()
    if args.cmd == "add-user":
        pw = args.password or getpass.getpass("Password: ")
        if len(pw) < 8:
            raise SystemExit("password must be at least 8 characters")
        db.run("INSERT INTO users (username, pw_hash, role, created) VALUES (?,?,?,?)",
               args.username, db.hash_pw(pw), args.role, time.time())
        print(f"created {args.role} {args.username}")
    elif args.cmd == "worker-token":
        from .app import WORKER_TOKEN
        print(WORKER_TOKEN)


if __name__ == "__main__":
    main()
