#!/usr/bin/env python3
"""Build an executed .ipynb from a jupytext-style 'cells' file.

Usage:  nb_build.py <cells.py> <out.ipynb> [--timeout SEC]

Cell syntax:
    # %%            -> code cell
    # %% [markdown] -> markdown cell
"""
import json
import sys
import pathlib


def parse(path: pathlib.Path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if text.strip():
            cells.append((kind, text))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "markdown" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    src = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    timeout = 3600
    if "--timeout" in sys.argv:
        timeout = int(sys.argv[sys.argv.index("--timeout") + 1])

    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(t) if k == "markdown" else nbformat.v4.new_code_cell(t)
        for k, t in parse(src)
    ]
    nb.metadata["kernelspec"] = {
        "name": "python3", "display_name": "Python 3", "language": "python"
    }
    out.parent.mkdir(parents=True, exist_ok=True)

    import os
    client = NotebookClient(
        nb, timeout=timeout, kernel_name=os.environ.get("NB_KERNEL", "python3"),
        resources={"metadata": {"path": str(out.parent)}},
        allow_errors=False,
    )
    client.execute()

    nbformat.write(nb, out)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    n_err = sum(
        1 for c in nb.cells if c.cell_type == "code"
        for o in c.get("outputs", []) if o.get("output_type") == "error"
    )
    print(f"wrote {out}  ({len(nb.cells)} cells, {n_code} code, {n_err} errors)")


if __name__ == "__main__":
    main()
