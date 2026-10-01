"""D1 applied for real: script IDs demoted from narrative subjects to provenance labels.

The rule that was supposed to hold is: a script name may appear as a citation of where a number
came from ("(Script: b11_theorem_remainder.py)"), but must not be the grammatical subject or
possessor of a sentence ("b48 adds a fifth", "b45's 30-88-89 family"), because that reads as a lab
notebook rather than as a paper.

Every replacement below is asserted to match exactly once, and the script refuses to write if any
one of them does not. Nothing is rewritten by regex over the whole file.
"""
import io, sys

PATH = "E:/pdf/topics/out/rc/chapter4.tex"

# (old, new) -- old must be a single-line fragment present exactly once.
EDITS = [
    # ---- Remark "what the chord-surrogate tests measured"
    (r"\texttt{b48} adds a fifth that searches rather than samples.",
     r"a fifth test (\texttt{b48}) searches rather than samples."),
    (r"\emph{One:} the flips of \texttt{b45} must go.",
     r"\emph{One:} the flips found by the two-cosine scan (\texttt{b45}) must go."),
    (r"($1600$ further draws in \texttt{b47}: $0$ resolved",
     r"($1600$ further draws in the capped rescan \texttt{b47}: $0$ resolved"),
    (r"cells (\texttt{b45}'s $30$-$88$-$89$ family) that run left unexplained; among them the cell of",
     r"cells (the $30$-$88$-$89$ family of that scan) that run left unexplained; among them the cell of"),
    (r"what the slope inflation put there. Along \texttt{b45}'s top-cosine ladder the log-log slope of",
     r"what the slope inflation put there. Along that scan's top-cosine ladder the log-log slope of"),
    (r"$\operatorname{cf}_1$, and \texttt{b48}, searching a $96\times240$ polar net instead of sampling,",
     r"$\operatorname{cf}_1$, and the polar-net search (\texttt{b48}, $96\times240$ directions) instead"),
    (r"direction). \texttt{b49} then sizes that descent in closed form, without an optimiser. On the steepest",
     r"direction). The closed-form sizing of that descent (\texttt{b49}) needs no optimiser. On the steepest"),
    (r"over-estimate at the median ($0.76$ worst), reproducing \texttt{b48}'s net range on a different seed by",
     r"over-estimate at the median ($0.76$ worst), reproducing the net's range on a different seed by"),
    # ---- Remark "what the side-bound test measured"
    (r"$\operatorname{cf}$ leaves behind is not an angle effect at all.} Script \texttt{b42} prescribes",
     r"$\operatorname{cf}$ leaves behind is not an angle effect at all.} The prescribed-spectrum run \texttt{b42} prescribes"),
    (r"rescaled $t$, is still within $3.4\%$ of $\lambda_{\min}(\Gr)$ --- and \texttt{b44}'s $2\times5$",
     r"rescaled $t$, is still within $3.4\%$ of $\lambda_{\min}(\Gr)$ --- and the $2\times5$"),
    (r"design settles \emph{which} hypothesis that residual is paying for.",
     r"design (\texttt{b44}) settles \emph{which} hypothesis that residual is paying for."),
    (r"\texttt{b45} says of what: by Proposition~\ref{prop:side} the residual is a \emph{moment defect}, the",
     r"the two-cosine scan says of what: by Proposition~\ref{prop:side} the residual is a \emph{moment defect}, the"),
    (r"(retraction~(m) of Section~\ref{sec:notclaims}): with the geometry held multi-atom, \texttt{b45}",
     r"(retraction~(m) of Section~\ref{sec:notclaims}): with the geometry held multi-atom, the same scan"),
    (r"\texttt{b44}'s tally is a statement about size, not about which hypothesis licenses a flip:",
     r"That design's tally is a statement about size, not about which hypothesis licenses a flip:"),
    (r"allows. \texttt{b44} also closes the audit",
     r"allows. The anisotropy design \texttt{b44} also closes the audit"),
    (r"\texttt{b42} left open: the smallest violating gap in the two flipping cells is",
     r"the prescribed-spectrum run left open: the smallest violating gap in the two flipping cells is"),
    (r"\emph{(iii) The root find is accurate in absolute terms, and \texttt{b43} measures where that stops",
     r"\emph{(iii) The root find is accurate in absolute terms, and the conditioning scan measures where that stops"),
    (r"the source of the order-$100$ errors of \texttt{b43}'s first scan, and the guarded endpoint restores",
     r"the source of the order-$100$ errors of the first version of that scan, and the guarded endpoint restores"),
    # ---- elsewhere in the body
    (r"not a technicality: in the two-arm least-squares experiment of the collaboration log (design",
     r"not a technicality: in the two-arm least-squares experiment behind this note (design"),
    (r"$S^1$ ($4000$ points) beats $\mu(w_0)$ in $157$ of the $840$ dictionaries of \texttt{b37}, and the",
     r"$S^1$ ($4000$ points) beats $\mu(w_0)$ in $157$ of the $840$ dictionaries tested in \texttt{b37}, and the"),
    (r"Proposition~\ref{prop:side} was written after \texttt{b44} left the size of the residual attributed",
     r"Proposition~\ref{prop:side} exists because the size of the residual had been attributed by \texttt{b44}"),
    (r"$1-t_1\mu>0$, and \texttt{b55} verifies each on every draw. Finally",
     r"$1-t_1\mu>0$, and the ray scan \texttt{b55} verifies each on every draw. Finally"),
    (r"finite-sample version was verified in \S19 of the collaboration log, script \texttt{b11}),",
     r"finite-sample version was verified numerically (script \texttt{b11}),"),
    (r"\texttt{b41} decides the question rather than the arithmetic: $c_1$ at $q=1,2,3,4,6$ is",
     r"the $q$-scan \texttt{b41} decides the question rather than the arithmetic: $c_1$ at $q=1,2,3,4,6$ is"),
    # ---- the retraction ledger of Section 8: keep every ID, attach a descriptor on first mention
    (r"aggregation, settled by the $q$-scan of \texttt{b41} rather than by arithmetic; Corollary~\ref{cor:dilution}",
     r"aggregation, settled by the $q$-scan \texttt{b41} rather than by arithmetic; Corollary~\ref{cor:dilution}"),
    (r"\texttt{b43} measures the law $\mathrm{relerr}\times\lambda_{\min}(\Gr)\approx1.7\cdot10^{-15}$ over six",
     r"the conditioning scan \texttt{b43} measures the law $\mathrm{relerr}\times\lambda_{\min}(\Gr)\approx1.7\cdot10^{-15}$ over six"),
    (r"$\operatorname{cf}-\lambda_{\min}$ sign flip to ``the angle distribution''. \texttt{b44} holds the",
     r"$\operatorname{cf}-\lambda_{\min}$ sign flip to ``the angle distribution''. The anisotropy design \texttt{b44} holds the"),
    (r"substantial cosine''. \texttt{b45} fixes the geometry and runs the two cosines as independent axes ---",
     r"substantial cosine''. The two-cosine scan \texttt{b45} fixes the geometry and runs the two cosines as independent axes ---"),
    (r"(n) the reading of \texttt{b47}'s family scan as evidence that no direction beats",
     r"(n) the reading of the family scan \texttt{b47} as evidence that no direction beats"),
    (r"$\operatorname{cf}_{1}$. \texttt{b48} searched instead of sampling --- a polar net of $96\times240$",
     r"$\operatorname{cf}_{1}$. The polar-net search \texttt{b48} replaced sampling --- a net of $96\times240$"),
    (r"artifacts of my own are retracted with it: the first \texttt{b48} took $v_{1}$ to be an arbitrary",
     r"artifacts of my own are retracted with it: the first version of that search took $v_{1}$ to be an arbitrary"),
    (r"it the acceptance thresholds \texttt{b51} was run to test. Its $\min_{\rm net}\lambda_1=0.0000$ in",
     r"it the acceptance thresholds tested by the direction-net run \texttt{b51}. Its $\min_{\rm net}\lambda_1=0.0000$ in"),
    (r"guarantee. \texttt{b55} falsified the claim exactly as it had been falsified-in-advance: on new tables",
     r"guarantee. The ray scan \texttt{b55} falsified the claim exactly as it had been falsified-in-advance: on new tables"),
    (r"concerns the closed form only. And \texttt{b42} put a price on the other relaxation one might reach",
     r"concerns the closed form only. And the prescribed-spectrum run \texttt{b42} put a price on the other relaxation one might reach"),
    (r"wrong (retractions~(l) and (m) below): \texttt{b44}'s $2\times5$ design attributes both effects to the",
     r"wrong (retractions~(l) and (m) below): the $2\times5$ design \texttt{b44} attributes both effects to the"),
    (r"(Corollary~\ref{cor:iso}) --- and \texttt{b45} names the anisotropy: the sign of the gap is the sign of",
     r"(Corollary~\ref{cor:iso}) --- and the two-cosine scan \texttt{b45} names the anisotropy: the sign of the gap is the sign of"),
    (r"the direction that matters for practice while staying open as a closed form: \texttt{b48} settled the",
     r"the direction that matters for practice while staying open as a closed form: the polar-net search \texttt{b48} settled the"),
    (r"\emph{negative} (retraction~(n)), and \texttt{b49} sized what the negative costs and buys. Along the",
     r"\emph{negative} (retraction~(n)), and the ray computation \texttt{b49} sized what the negative costs and buys. Along the"),
    (r"direction \texttt{b50} evaluated, $\lambda_2(v)\le\lambda_1(v)$; the pointwise ordering of successive",
     r"direction tested by \texttt{b50}, $\lambda_2(v)\le\lambda_1(v)$; the pointwise ordering of successive"),
    (r"the gap). \texttt{b51}--\texttt{b53} then settle the direction half of that alternative, and settle it",
     r"the gap). The three direction runs \texttt{b51}--\texttt{b53} then settle the direction half of that alternative, and settle it"),
    (r"$5\cdot10^{-4}$ (Remark~\ref{rem:tight}). \texttt{b55} then replaces the trade-off that these runs",
     r"$5\cdot10^{-4}$ (Remark~\ref{rem:tight}). The ray scan \texttt{b55} then replaces the trade-off that these runs"),
    (r"\texttt{b59} narrows it without closing it --- on $264$ rays through $v_1$, minimised exactly and",
     r"The leverage gate \texttt{b59} narrows it without closing it --- on $264$ rays through $v_1$, minimised exactly and"),
    (r"neither $v_\ast$ nor that floor. \texttt{b60} then replaced that sampling by the exact sphere gradient of",
     r"neither $v_\ast$ nor that floor. The cross-lane run \texttt{b60} then replaced that sampling by the exact sphere gradient of"),
    (r"\texttt{b50} sharpens that caveat in the unfavourable direction and I record it here rather than only in",
     r"The two-atom check \texttt{b50} sharpens that caveat in the unfavourable direction, and it is recorded here rather than only in"),
]


def main():
    text = io.open(PATH, encoding="utf-8", newline="").read()
    out = text
    changed = []
    for old, new in EDITS:
        n = out.count(old)
        assert n == 1, (n, old[:70])
        assert old != new
        out = out.replace(old, new, 1)
        changed.append((old, new))

    # invariants: no theorem/remark environment may be disturbed, and every script ID that the body
    # named before must still be named at least once in the body -- a dropped mention is allowed only
    # where a first mention with the same ID survives nearby (checked here as set equality).
    for tok in ("\\begin{theorem}", "\\begin{remark}", "\\end{document}", "\\begin{thebibliography}"):
        assert out.count(tok) == text.count(tok), tok
    cut = "\\section{Reproduction}"
    b0, b1 = text.index(cut), out.index(cut)
    import re
    pat = re.compile(r"\\texttt\{(b\d+[a-z]?)")
    before, after = set(pat.findall(text[:b0])), set(pat.findall(out[:b1]))
    assert before == after, ("IDs lost from the body: %s / gained: %s" % (sorted(before - after), sorted(after - before)))
    print("body script-ID set unchanged (%d ids); mentions %d -> %d"
          % (len(after), len(pat.findall(text[:b0])), len(pat.findall(out[:b1]))))

    io.open(PATH, "w", encoding="utf-8", newline="").write(out)
    print("applied %d edits" % len(changed))
    for old, new in changed:
        print("  -", old[:64])
        print("  +", new[:64])


if __name__ == "__main__":
    main()
