"""
Koba Document Scanner — main.py
=================================
Unified launcher.

Usage:
    python main.py app          # starts the web dashboard (default)
    python main.py hotfolder    # starts the hot folder watcher
    python main.py hotfolder --folder D:/scan --interval 10

Run  python main.py --help  for full options.
"""

import sys


def main():
    # Default to 'app' if no subcommand is given
    if len(sys.argv) < 2 or sys.argv[1] not in ("app", "hotfolder", "--help", "-h"):
        sys.argv.insert(1, "app")

    mode = sys.argv[1]

    if mode == "app":
        # Remove the subcommand so app.py's arg parsing isn't confused
        sys.argv.pop(1)
        import app as _app  # noqa: F401 — runs __main__ block via __name__ guard

        import os
        from pathlib import Path

        port = int(os.environ.get("PORT", 5000))
        print("")
        print("  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        print("  ┃  🏛️  Koba Document Scanner              ┃")
        print(f"  ┃  ➡  http://localhost:{port:<31}┃")
        print("  ┃  📂  Uploads stored in: uploads/          ┃")
        print("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        print("")

        _app.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

    elif mode == "hotfolder":
        sys.argv.pop(1)  # remove 'hotfolder', leave remaining args for hot_folder.main()
        from hot_folder import main as hf_main
        hf_main()

    else:
        print(__doc__)
        sys.exit(0)


if __name__ == "__main__":
    main()
