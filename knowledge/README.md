# The knowledge base

Everything this repository has read about evaluating LLM applications, as one file per
topic, for agents and people to query. The query surface is [`llms.txt`](../llms.txt)
at the repository root: fetch it, pick a line, fetch that file.

## Shelves

- `methodology/`: what the practitioners whose work `axial` implements say to do, and
  where they disagree. Sources are their writing. Changes rarely.
- `procedures/`: a loop a team runs end to end, written as numbered steps, each with its
  trigger, its artifact and whether a human gates it. Cites landscape entries for the
  tools it uses and methodology entries for the numbers it quotes.
- `landscape/`: one tool per file. Claims decay, so every file carries the date its
  sources were read and the index shows it.

## Frontmatter

Every file starts with YAML frontmatter. Required: `name` (equals the filename stem,
kebab-case, unique across shelves), `description` (one line, at most 160 characters, no
trailing period; it is the index line, so say what the file answers), `shelf` (equals
the directory), `updated` (date of the last substantive change), `sources` (a list of
`{url, read}`). Optional: `related` (names of other files), and for landscape entries
`license` (SPDX identifier, or `proprietary`) and `extra` (the optional-dependency
group in `pyproject.toml` through which `axial` depends on the tool; absent means no
dependency).

Links between files are relative Markdown links so GitHub renders them. The check
resolves them.

Landscape entries, except `landscape-overview`, have the same seven headings in this
order: What it is, Core abstraction, Where it sits, Maturity, Biggest complaint,
Relation to axial, Corrections. A heading with nothing to say holds one line.
Corrections are dated notes, newest last; a corrected claim stays in the file.

## Commands

```bash
uv run python scripts/knowledge_index.py            # rewrite llms.txt
uv run python scripts/knowledge_index.py --check    # validate; CI runs this
```

The check reports every violation before exiting and never writes an index for a
broken tree. The rules above are the reader's version of
[`docs/superpowers/specs/2026-10-07-knowledge-base-layout-design.md`](../docs/superpowers/specs/2026-10-07-knowledge-base-layout-design.md);
when the schema changes, both change in the same commit.
