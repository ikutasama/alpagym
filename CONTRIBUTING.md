# Contributing to AlpaGym

## Living Documents

This document, as well as the coding style in AGENTS.md, is a living style guide
for the codebase. It is not meant to be a comprehensive set of rules, but rather
a collection of principles and examples to guide the development of clean,
readable, and maintainable code. It will evolve over time as we learn and adapt
our practices.

**Update process:** When we encounter a disagreement about code style during a
review that cannot be resolved because i) a missing or unclear rule, or ii)
because we believe the rule should be updated, we pose the question to the team
through an *async Slack thread*.
If no agreement can be reached in the thread, Max is the tie breaker.

## On the use of AI.

There is a difficult trade-off between using AI tools to speed up development
and maintaining human oversight to ensure code quality and familiarity with the
codebase. As models and our practises evolve, the sweet spot on this trade-off
will likely shift.

While we acknowledge that code quality and human readability might become less
important as models improve, for now we still aim for high quality, clean,
human-readable code, despite its costs, because:

- We believe it benefits both human and AI readers.
- We place a high value on being familiar with the codebase and understanding it
  deeply, as subtle bugs in networking and RL algorithms can be costly but hard
  to detect.
- Starting with a clean architecture and code base will make it easier to
  maintain and adapt as we learn.
- We aim for a high quality bar for the AlpaX ecosystem to distinguish us from
  other existing frameworks.

## Review Process

Following from the motivations outlined in the previous sections, we aim for a
review process that still requires human oversight and judgement.
At the same time, we want to test and experiment how much we can delegate to AI
tools without sacrificing code quality.

Guidelines:

- Two human reviews per MR (in addition to the automated AI reviews).
- Before requesting review, the MR author should have looked at every changed
  line outside generated or lock files.
- One code owner review, one non-code owner review. Both are careful human
  reviews, meaning that while AI tools can be used to assist, the human reviewer
  should have looked at every line of code (excluding tests).
- We assign code-owners to different parts of the codebase, but we also
  encourage non-code owners to review MRs in other parts of the codebase to
  increase familiarity across the team.
- Reviewers can ask for MRs to be reduced in scope if they are too large to
  review effectively.

**Important:** Be conscious of how well the AI review aligns with your own
judgement. We want to improve our understanding of what is and isn't possible
with AI and over time improve our processes to rely more heavily on AI.

## Coding principles

Coding principles are outlined in @AGENTS.md.

## Signing Your Work

We require that all contributors "sign off" on their commits. This certifies
that the contribution is your original work, or you have rights to submit it
under the same license, or a compatible license.

- Any contribution which contains commits that are not signed off will not be
  accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when
committing your changes:

```bash
$ git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

Full text of the DCO:

```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
  1 Letterman Drive
  Suite D4700
  San Francisco, CA, 94129

  Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
```

```
  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) the contribution was created in whole or in part by me and I have the right to submit it under the open
      source license indicated in the file; or

  (b) the contribution is based upon previous work that, to the best of my knowledge, is covered under an
      appropriate open source license and I have the right under that license to submit that work with
      modifications, whether created in whole or in part by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated in the file; or

  (c) the contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have
      not modified it.

  (d) I understand and agree that this project and the contribution are public and that a record of the
      contribution (including all personal information I submit with it, including my sign-off) is maintained
      indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```
