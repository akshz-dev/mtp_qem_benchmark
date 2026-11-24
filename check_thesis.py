r"""Structural validator for the thesis source tree.

Checks what a compiler would complain about, without needing a compiler:
  - every \subfile / \include / \input target exists
  - every \includegraphics path resolves
  - every \ref / \eqref / \cite target is defined somewhere
  - labels are unique
  - braces and $ are balanced per file
  - subfiles carry the required header and no stray \end{document} in \include'd files

Usage: python check_thesis.py   (run from the project root)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
problems, notes = [], []


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def tex_files():
    out = []
    for root, _, fs in os.walk(ROOT):
        if ".git" in root:
            continue
        for f in fs:
            if f.endswith(".tex"):
                out.append(os.path.relpath(os.path.join(root, f), ROOT))
    return sorted(out)


files = tex_files()
src = {f: strip_comments(open(os.path.join(ROOT, f), encoding="utf-8").read()) for f in files}
raw = {f: open(os.path.join(ROOT, f), encoding="utf-8").read() for f in files}

# ---- 1. inclusion graph -----------------------------------------------------
included = set()
for f, s in src.items():
    for cmd in ("subfile", "include", "input"):
        for t in re.findall(r"\\%s\{([^}]+)\}" % cmd, s):
            t = t.strip()
            cand = [t, t + ".tex"]
            if not any(os.path.exists(os.path.join(ROOT, c)) for c in cand):
                problems.append(f"{f}: \\{cmd}{{{t}}} -> no such file")
            else:
                included.add(next(c for c in cand
                                  if os.path.exists(os.path.join(ROOT, c))).replace(".tex", "") + ".tex")

orphans = [f for f in files if f not in included and f != "Thesis.tex"]
if orphans:
    notes.append("not included from Thesis.tex: " + ", ".join(orphans))

# ---- 2. graphics ------------------------------------------------------------
for f, s in src.items():
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s):
        if not any(os.path.exists(os.path.join(ROOT, g + e))
                   for e in ("", ".png", ".jpg", ".jpeg", ".pdf")):
            problems.append(f"{f}: \\includegraphics{{{g}}} -> missing image")

for f, s in src.items():
    for g in re.findall(r"\\lstinputlisting(?:\[[^\]]*\])?\{([^}]+)\}", s):
        if not os.path.exists(os.path.join(ROOT, g)):
            problems.append(f"{f}: \\lstinputlisting{{{g}}} -> missing file")

# ---- 3. labels and references ----------------------------------------------
labels, dup = {}, []
for f, s in src.items():
    for l in re.findall(r"\\label\{([^}]+)\}", s):
        if l in labels:
            dup.append(f"{l} (in {labels[l]} and {f})")
        labels[l] = f
for d in dup:
    problems.append("duplicate label: " + d)

for f, s in src.items():
    for r in re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", s):
        if r not in labels:
            problems.append(f"{f}: \\ref{{{r}}} -> undefined label")

# ---- 4. citations ----------------------------------------------------------
bibkeys = set()
for root, _, fs in os.walk(ROOT):
    for f in fs:
        if f.endswith(".bib"):
            bibkeys |= set(re.findall(r"@\w+\{([^,]+),",
                                      open(os.path.join(root, f), encoding="utf-8").read()))
for f, s in src.items():
    for m in re.findall(r"\\cite[pt]?\{([^}]+)\}", s):
        for k in m.split(","):
            if k.strip() and k.strip() not in bibkeys:
                problems.append(f"{f}: \\cite{{{k.strip()}}} -> not in any .bib")

# ---- 5. balance and subfile hygiene ----------------------------------------
for f, s in src.items():
    op = len(re.findall(r"(?<!\\)\{", s)) - len(re.findall(r"(?<!\\)\}", s))
    if op:
        problems.append(f"{f}: brace imbalance ({op:+d})")
    if len(re.findall(r"(?<!\\)\$", s)) % 2:
        problems.append(f"{f}: odd number of unescaped $")
    for env in ("longtable", "tabular", "figure", "table", "itemize", "enumerate",
                "equation", "align", "document"):
        b = len(re.findall(r"\\begin\{%s\*?\}" % env, s))
        e = len(re.findall(r"\\end\{%s\*?\}" % env, s))
        if b != e:
            problems.append(f"{f}: \\begin/\\end{{{env}}} mismatch ({b} vs {e})")

for f in files:
    if f.startswith("chapters/chapter") or f == "chapters/chapter2.tex":
        if "\\documentclass[Thesis.tex]{subfiles}" not in raw[f]:
            problems.append(f"{f}: missing \\documentclass[Thesis.tex]{{subfiles}} header")
        if "\\end{document}" not in raw[f]:
            problems.append(f"{f}: subfile without \\end{{document}}")
    if f.startswith("appendices/"):
        if "\\end{document}" in raw[f]:
            problems.append(f"{f}: \\include'd file must NOT contain \\end{{document}}")

# ---- report ----------------------------------------------------------------
print(f"checked {len(files)} .tex files, {len(labels)} labels, {len(bibkeys)} bib entries")
for n in notes:
    print("note:", n)
if problems:
    print(f"\n{len(problems)} PROBLEM(S):")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("\nno structural problems found")
