# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| < 0.1 | No |

Security fixes are released for the latest minor version only.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it
privately through GitHub's private vulnerability reporting:
[Security → Report a vulnerability](https://github.com/Ahsan-muhd-444/house-price-prediction/security/advisories/new).

Include what is affected, how to reproduce it, and its impact as you see it.
You should receive an acknowledgement within 7 days and an assessment within
30. Once a fix is released, the advisory is published with credit to you,
unless you prefer otherwise.

## Things to know

- **Model files are pickles.** `load_model`, `amesml evaluate`, `amesml predict`
  and `amesml serve` unpickle `model.joblib`, which can execute arbitrary code.
  Only load models that you, or someone you trust, created. The SHA-256 check
  against `metadata.json` detects corrupted or swapped files, but an attacker
  who can replace one file can replace both.
- **The REST API has no authentication or rate limiting.** Run it behind your
  own gateway or inside a private network, not directly on the public internet.
- **Dataset downloads are checksummed.** `amesml data download` refuses a file
  whose SHA-256 does not match the pinned value, unless you pass a custom
  `--url` without `--sha256`.
