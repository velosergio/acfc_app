import sys


def _main() -> int:
    argv = sys.argv[1:]
    if not argv or "--gui" in argv:
        from acfc.gui import run_gui

        return run_gui()
    from acfc.cli import main

    return main(argv)


if __name__ == "__main__":
    sys.exit(_main())
