# References / Referencias

The bibliography cited in the article, exported in three interchangeable
formats. All three contain the **same 63 entries** and are kept in sync.

La bibliografía citada en el artículo, exportada en tres formatos
intercambiables. Los tres contienen las **mismas 63 entradas** y se mantienen
sincronizados.

| File | Format | Use with |
|---|---|---|
| `bpm_references_bibtex.bib` | Classic **BibTeX** (`journal`, `year`, `address`) | `\bibliography{}` with `bibtex`; Overleaf default; most journal templates (Elsevier, IEEE, Springer) |
| `bpm_references_biblatex.bib` | **BibLaTeX** (`journaltitle`, `date`, `location`) | `\addbibresource{}` with `biblatex` + `biber`; richer date and localisation handling |
| `bpm_references_zotero.rdf` | **Zotero RDF** (XML) | Zotero, Mendeley, JabRef — import via *File → Import…*; preserves item types, abstracts and attachments |

## Usage / Uso

**BibTeX**

```latex
\bibliographystyle{apalike}
\bibliography{references/bpm_references_bibtex}
```

**BibLaTeX** (compile with `biber`, not `bibtex`)

```latex
\usepackage[backend=biber,style=apa]{biblatex}
\addbibresource{references/bpm_references_biblatex.bib}
...
\printbibliography
```

**Zotero:** *File → Import…* → select `bpm_references_zotero.rdf`. The importer
creates a new collection with the 63 items.

## Coverage / Cobertura

The bibliography spans the three strands the article integrates: BPM theory and
capabilities (Hammer, Dumas, vom Brocke, Rosemann, van der Aalst, Trkman,
Škrinjar), PLS-SEM methodology (Hair, Henseler, Sarstedt, Shmueli, Becker,
Kock), the SEM–ANN hybrid tradition (Leong, Sharma, Luyao, Le) and set-theoretic
methods (Ragin, Fiss, Schneider & Wagemann, Pappas & Woodside, Dul).

La bibliografía cubre las tres líneas que el artículo integra: teoría y
capacidades de BPM, metodología PLS-SEM, la tradición híbrida SEM–ANN y los
métodos de teoría de conjuntos.

## Notes / Notas

- Two pairs of entries are **content duplicates** with distinct citation keys:
  `harman_modern_1976` / `harman_modern_1976-1` (Harman, *Modern Factor
  Analysis*) and `LizanoMora2026BPMTriMetodologico` / `lizano-mora_bpm_2026`
  (this repository's self-citation). They do not collide as keys, so LaTeX
  compiles without error, but citing both would produce a duplicated entry in
  the reference list.
- `Becker2012HierarchicalLatent` carries `number = {5}` in the BibLaTeX export
  and `number = {5–6}` in the BibTeX one. The issue is 5–6.
- The self-citation of this repository is included so that the article can cite
  its own reproduction package; see [`../CITATION.cff`](../CITATION.cff).
