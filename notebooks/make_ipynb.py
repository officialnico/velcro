"""Convert the `# %%`-cell script into a Jupyter notebook (no jupytext dependency)."""
import json, re, sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text().split("\n")
cells, cur, kind = [], [], None
def flush():
    if cur and any(l.strip() for l in cur):
        text = "\n".join(cur).strip("\n")
        if kind == "markdown":
            text = "\n".join(re.sub(r"^# ?", "", l) for l in text.split("\n"))
        cells.append({"cell_type": kind, "metadata": {}, "source": text + "\n", **({"outputs": [], "execution_count": None} if kind == "code" else {})})
for line in src:
    m = re.match(r"^# %%( \[markdown\])?", line)
    if m:
        flush(); cur = []; kind = "markdown" if m.group(1) else "code"
    else:
        cur.append(line)
flush()
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
pathlib.Path(sys.argv[2]).write_text(json.dumps(nb, indent=1))
print("wrote", sys.argv[2], len(cells), "cells")
