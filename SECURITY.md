# Security policy

## Supported versions

Turbo-WinFare is a single-developer project with no release branches. Only the latest commit on
`main` is supported. Please confirm an issue reproduces there before reporting it.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Use GitHub's private reporting instead:
[Security → Report a vulnerability](https://github.com/justin-creator222/Turbo-WinFare/security/advisories/new).

Include the version or commit, your OS and GPU, and a reproduction. Expect an acknowledgement
within about a week; this is a spare-time project, so please be patient.

## Scope

Things worth reporting:

* Memory-safety bugs reachable from untrusted input — a malformed `.gturbo` bundle, a crafted
  `tokenizer.json`, or an HTTP request to the embedded server.
* Path traversal or arbitrary file read/write through the model loader, the GUI static file
  handler, or the conversion script.
* Anything that lets a request to the local server escape the intended request handling.

Known limitations that are **not** vulnerabilities:

* **The embedded HTTP server has no authentication and is meant for localhost only.** Binding it
  to a reachable interface exposes unauthenticated model inference to anyone who can reach the
  port. Do not do that without putting your own authenticating proxy in front of it.
* Runtime integrity verification of bundle files and symlink rejection are **not yet
  implemented** — a `.gturbo` bundle is trusted input. Only load bundles you produced yourself
  with `tools/convert_hf_to_gturbo.py`.
* Resource exhaustion from a legitimately large prompt or context. Generation is serialized and
  the queue is bounded by `--queue-limit`, but a single request can still be slow.
* Issues in the model's *output* — hallucination, unsafe content — are a property of Gemma 4, not
  of this engine.

Vulnerabilities in the upstream Swift/Metal implementation belong to
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare), not here.
