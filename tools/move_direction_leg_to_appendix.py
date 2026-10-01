"""Relocate the direction-leg ledger out of the body of chapter4.tex into an appendix.

Why a script and not hand-editing: the block is 315 lines of dense displayed mathematics
(1096-1410, Remark rem:tight plus Remark rem:gauss). Retyping it through the editing tool
would risk silent symbol corruption, which is worse for the manuscript than leaving it in
place. This script therefore MOVES the block byte-for-byte and asserts that before writing
anything: it re-extracts the block from the output file and compares it with the input bytes,
and it checks the body no longer contains it. The only prose it introduces is the 3-sentence
pointer remark (rem:dirleg) and the appendix heading, both written below verbatim.
"""
import io
import os
import sys

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chapter4.tex")
P = os.path.normpath(P)

BEGIN = 1096  # 1-indexed: \begin{remark}[what b51--b55 measured ...]  (rem:tight)
END = 1410    # 1-indexed: \end{remark} closing rem:gauss

with io.open(P, "r", encoding="utf-8", newline="") as f:
    text = f.read()

nl = "\r\n" if "\r\n" in text else "\n"
lines = text.split(nl)

block = lines[BEGIN - 1:END]
assert block[0].startswith("\\begin{remark}") and "b51" in block[0], repr(block[0])
assert r"\label{rem:tight}" in lines[BEGIN], lines[BEGIN]
assert r"\begin{remark}" in lines[1369] and r"\label{rem:gauss}" in lines[1370], lines[1369:1372]
assert block[-1].strip() == r"\end{remark}", repr(block[-1])
assert r"\end{document}" in lines[-1] or any(l.strip() == r"\end{document}" for l in lines)

enddoc = [i for i, l in enumerate(lines) if l.strip() == r"\end{document}"]
assert len(enddoc) == 1, enddoc
i_end = enddoc[0]
# the block must precede \end{document}
assert END - 1 < i_end, (END, i_end)

POINTER = [
    r"\begin{remark}[What the direction leg did not buy]",
    r"\label{rem:dirleg}",
    r"The direction leg was run to look for a cheap way to close the residual gap",
    r"$\operatorname{cf}_1-\lambda_{\min}(\Gr)$ by optimising the direction $v$ alone, and it bought",
    r"nothing: the $n=1$ chord loses $0.50$--$1.01$ of that gap \emph{even at} $v_\ast$, the direction",
    r"that attains the exact fibre crossing of Proposition~\ref{prop:tight}, whereas one additional",
    r"scalar moment removes $1.00$ of it on the same tables (median \emph{and} max, $20/20$ cells).",
    r"The residual of $\operatorname{cf}_1$ is therefore truncational, not directional, and the",
    r"angle $\theta=\angle(v_\ast,v_b)$ that a reader might otherwise take for a convergence error is",
    r"a model-discrepancy angle: $v_\ast$ is not a critical point of $\lambda_1$ at all, so no",
    r"stationary-point story explains $\operatorname{cf}_1$. The measurement is recorded in full in",
    r"Appendix~\ref{app:direction} --- direction nets and their resolution limits, the two independent",
    r"routes to the descent-set boundary, the cubic-share attribution of the residual, and the moment",
    r"hierarchy --- because a negative result is only as credible as the resolutions at which it was",
    r"tested.",
    r"\end{remark}",
]

APX_HEAD = [
    r"\appendix",
    r"",
    r"\section{The direction leg, in full}\label{app:direction}",
    r"This appendix is the measurement ledger behind Remark~\ref{rem:dirleg}; it is kept complete",
    r"rather than summarised because every figure in it is what retired a claim of mine, and because",
    r"the script identifiers are the pointers used by Section~\ref{sec:reproduce}.",
    r"",
]

out = lines[: BEGIN - 1] + POINTER + lines[END:i_end] + APX_HEAD + block + [""] + lines[i_end:]

new = nl.join(out)

# --- assertions on the result, before writing ---
assert new.count(block[0]) == 1, "moved remark must appear exactly once"
# the block is MOVED, not copied, so remark bookkeeping must change by exactly the
# one pointer remark that replaces it
for env in ("\\begin{remark}", "\\end{remark}"):
    assert new.count(env) == text.count(env) + 1, (env, text.count(env), new.count(env))
# the moved block must appear verbatim, contiguously, in the output
i2 = out.index(block[0])
assert out[i2: i2 + len(block)] == block, "block not byte-identical after move"
assert i2 > out.index(r"\appendix"), "block must sit after \appendix"
# body: between the pointer and \appendix there must be no rem:tight / rem:gauss text
i_ptr = out.index(r"\label{rem:dirleg}")
i_apx = out.index(r"\appendix")
body = nl.join(out[i_ptr:i_apx])
for probe in (r"\label{rem:tight}", r"\label{rem:gauss}"):
    assert probe not in body, probe
assert len(out) == len(lines) - len(block) + len(POINTER) + len(APX_HEAD) + len(block) + 1

with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write(new)

print("moved %d lines (source %d-%d) -> appendix; body lines %d -> %d"
      % (len(block), BEGIN, END, len(lines), len(out)))
print("block bytes: %d, identical in output: True" % len(nl.join(block)))
