# ADR 0001: License and repository visibility (D-19)

- Status: Accepted
- Date: 2026-07-02
- Decided by: Melvin Hoyer III (project author)

## Context

PRD decision D-19 left the license and repository visibility open, to be
resolved before the first substantive public push. The choice matters because
a public commit of substantive technical content is a legal disclosure event:
it starts the 12-month United States grace period for any patentable subject
matter it discloses, and it immediately destroys patent rights in
absolute-novelty jurisdictions. The candidate outcomes were MIT, Apache-2.0,
or an all-rights-reserved private repository.

## Decision

The project is licensed under the Apache License, Version 2.0, and the
repository remains public at https://github.com/JusHoya/star_reacher.

## Rationale

- Apache-2.0 carries an explicit patent grant and a corresponding
  patent-retaliation clause, which MIT lacks. For a project whose author
  maintains an active patent strategy elsewhere, the explicit grant makes the
  licensing position of contributions and use unambiguous rather than implied.
- The product requirements document is already publicly visible in this
  repository, so disclosure of the design has effectively begun; withdrawing
  to a private repository would not restore novelty for what is already
  published, and continuing publicly keeps the record consistent.
- The project's publication and portfolio goals (a citable repository,
  a math-library PDF, and a scientific report carrying the author's byline)
  require a public repository and a recognized open-source license to be
  meaningful.

## Consequences

- Every public commit from this point is a disclosure: the 12-month United
  States grace clock runs on any patentable subject matter a commit reveals,
  and foreign patent rights in absolute-novelty jurisdictions are forfeited
  for disclosed material. Anything intended for patent protection must be
  filed before it is committed here.
- The full Apache-2.0 text lives in `LICENSE` at the repository root;
  packaging metadata declares `license = "Apache-2.0"`; `CITATION.cff`
  carries the matching SPDX identifier. The README license badge and section
  reflect the decision.
- The development plan is unchanged; D-19 anticipated proceeding identically
  under any of the candidate outcomes.
