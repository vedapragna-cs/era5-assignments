# Manifest

Every file in this submission with its SHA-256. Regenerate with `python3 manifest.py` and verify any single file with `sha256sum <path>`.

18 files, 486,873 bytes total. Generated 2026-08-22.


> **One file is deliberately volatile.** `data/date-verification.json` carries a
> `checked_at` timestamp and is rewritten every time `verify_dates.py` runs, so its hash
> below is the value as committed. Running the check will change it; that is the check
> working, not the artifact drifting. Every other hash here is stable.


## Pages (committed, no build step needed)

| file | bytes | sha256 |
|---|---:|---|
| `corrections.html` | 26,962 | `100dc10178d4b05f78c8d806406b1cc2872f84ab38810ece1a200c98994a7c2b` |
| `findings.html` | 41,646 | `0ceea2b19ded98d1123f84ae3104fd2fb0491381e2d3f36303c1c20196e0f5b1` |
| `index.html` | 114,245 | `bb84591282c2ffaee550bf2bb62cc11d7361b20f89dc720f9bf0c1970c89bbef` |
| `sources.html` | 17,299 | `aa2fc1b798fdd0261a3ca00ce8021860052f35a54efcbb57e65e8349c746dd08` |

## Data — single source of truth

| file | bytes | sha256 |
|---|---:|---|
| `data/arxiv-response.xml` | 73,367 | `b3f09d86ed5f9562d1044a466a27c9fa64f12dd11e9548c7e15ab0d4934fe831` |
| `data/corrections.json` | 16,881 | `c2316e605f160c581547ace68c7bb7cc685eb4f5c7207a8f448d941022cd8f81` |
| `data/date-verification.json` | 5,472 | `4e4617529d68306320d0ff8d7bf6c7ae5397ccfd04d886a93fc2db36fe196ffb` |
| `data/findings.json` | 23,181 | `0219822866581c5c8cedbf2ea2191cdc223a2763b5b06c6be8bae85ea3177a88` |
| `data/mechanisms.json` | 76,531 | `1327f2597000b1c340476ea43b1765d7c834217851197add39ff8eb5bf224ee3` |

## Scripts — the checks

| file | bytes | sha256 |
|---|---:|---|
| `analyse.py` | 2,368 | `4220f4c59bfe1962286d8233b2c9e393b7c9ba0d25b68d5640e832911708b09c` |
| `build.py` | 26,670 | `9396cb827b4f86b354265fb72da51d5778aaf5dc602eeeabd2b7fc59cb8f5450` |
| `charts.py` | 11,984 | `15b858f94f419beb7e35696985a817c59e73dc911a6ee479b33354c2b8e537fc` |
| `check_consistency.py` | 4,473 | `b4817fc308df00da99f66aacc6fe78aed2a3b3dacaebbf15bcc08a415b5898d0` |
| `manifest.py` | 2,439 | `135d2685754399db10f202a4c74cfc41c1bd8f7d3ad1fd82b9fcb1c943ce3c71` |
| `verify_dates.py` | 3,551 | `601c0b8de5bd8ddd13fdd7f91cd68213234e87a2d8d04f2ec0dda526e37660ef` |

## Documents

| file | bytes | sha256 |
|---|---:|---|
| `ARCHITECTURE.md` | 3,176 | `7fcb555f10f69a7f9710ae2d64ef462f8f43c5c30fc55e22084b497ab19cdb53` |
| `EXPERIMENT_LOG.md` | 24,643 | `a8cddc8299d0480a56d5860b628b509d81bbc299772eaecfaa7054a07642d962` |
| `README.md` | 11,985 | `8c1362f4a716c7dd36b7640fbdc38114de3f1a44a01e88232af18085c39b8b16` |
