# Gotcha — examshield

An offline CLI that turns an exam document on your machine (PDF, Word) into a copy with the same content that is harder for AI to read and solve, and proves the effect with numbers stamped with a date and model versions.

> **Status: early development (Phase 5 done, free scope implemented).**
> `protect` (makes the copies), `check` (checks the content is the same), `verify` and `report` (measure the effect) `trace` (finds whose copy leaked) and `profile` (lists profiles and their verification state) work. Cloud model clients (OpenAI, Anthropic, Google) are included but have never been called with a real key. There are no scores from a real AI model yet; the only numbers so far are the text-extraction and OCR baselines.
> The free scope (Phases 1 to 5) is fully built. No profile has been verified against a real AI model yet, so `protect` prints a warning for every `a-` profile. Paid plans are still in planning.

**Jump to:** [Quick start](#quick-start) · [**Usage**](USAGE.md) · [Installation](#installation) · [Roadmap](#roadmap) · [Spec](spec/README.md) · [한국어](README.md)

The usage guide, the spec and the CLI's own messages are in Korean.

## Overview

Cheating with smart glasses and generative AI has been caught repeatedly in exams in Korea. Every countermeasure in force today polices devices; we found none that works on the exam paper itself. This tool aims at that gap. The background is in [spec/02-background.md](spec/02-background.md).

It does three things:

1. It turns an exam document into a copy with the same content, adding text-layer scrambling, hidden decoys and a tracking watermark per examinee.
2. It checks by machine that the original and the copy have the same content.
3. It measures, per model, how well the copy resists being read and solved by AI, and writes a report.

**This tool does not promise that "AI cannot read it". It promises "on this date, against these models, the legibility was this many percent".** The effect of every technique is a hypothesis; only the ones that survive measurement are used.

It is for teachers, exam writers and assessment bodies who hand out online exams or assignment files.

## Features

| Feature | Status |
|---|---|
| Project skeleton: uv, ruff, mypy, pytest, checks before push (git pre-push hook) | Done |
| Dev-time safety hooks and read blocking for secret files (see [Safety hooks](#safety-hooks)) | Done |
| Spec split into `spec/` with a check that it still equals the original | Done |
| Built-in item set v1 (12 dummy items written for this project, not past exam papers) | Done |
| `examshield verify` — measures a profile on the built-in item set and writes a report | Done (control group and four scenario A profiles) |
| Capture simulation: resolution, skew, perspective, lighting, blur, noise, JPEG | Done (never compared with real photos) |
| Baselines: text-layer extraction, Tesseract OCR | Done (Tesseract was run with English data only) |
| Local model (Ollama) client | Done (tested against a fake server only, never run against a real Ollama) |
| `examshield report` — shows a report file as a table | Done |
| `examshield protect` — makes copies with scenario A (digital distribution) profiles: rasterizing, glyph mapping scrambling, hidden decoys | Done (Word input tested only with a stand-in converter, without LibreOffice) |
| `examshield check` — checks that the original and the copy have the same content: page count, page size, picture count, text, rendering, OCR | Done |
| `examshield auth` — keeps API keys in the OS keychain | Done (tested with a test keychain only, never written to a real OS keychain) |
| Cloud model verification (your own key, off by default), confirmation before calls, `--offline` | Done (each vendor's SDK tested against a fake server only, no real call was made) |
| Tracking watermark per examinee, `--copies`, `examshield trace` | Done (recovered from PDFs and screenshot-quality pictures; checked with PDFs rendered to pictures, not real screen captures) |
| Model drift warnings (`protect` warns when verification is missing, older than 30 days or below the bar), `examshield profile list\|show` | Done (no verification record from a real AI yet) |
| Scenario B (printed exam) profiles, watermark recovery from photos | Planned (paid) |
| Single executable for air-gapped networks, commercial license, verified profile subscription | Planned (paid) |

### How it works

- **Transforming needs no network**: `protect` and `check` make zero network calls. The tests block socket creation and check that both still pass.
- **The original stays untouched**: the original is opened read-only, and results always go to a different folder.
- **Same input, same output**: with the same input, profile and seed, the copy is identical byte for byte.
- **What gets verified is the profile, not your document**: `verify` applies the profile to the built-in public item set and measures that. A real exam paper is not sent to the cloud.
- **API keys are yours**: cloud verification is optional. Keys are read from environment variables or the OS keychain and never written to a plain-text file.
- **The effect expires**: a verification result is a measurement against one model on one day. `protect` looks up the profile's latest verification report and warns when there is none, when it is older than 30 days, or when it falls below the bar. A warning never blocks the copy.

All security requirements are in [spec/05-security.md](spec/05-security.md).

## Requirements

- Python **3.11 or newer** (developed on 3.12)
- [uv](https://docs.astral.sh/uv/)
- Git
- Optional: [Tesseract](https://github.com/tesseract-ocr/tesseract) with `kor` and `eng` data (OCR baseline), [Ollama](https://ollama.com) with a vision model (local model measurement), LibreOffice (Word input)

## Installation

The package is not published yet. Install from source:

```bash
git clone https://github.com/Trapa-Eureka/Gotcha.git
cd Gotcha
uv sync
```

The package is not on PyPI yet. Once it is, the plan is to make this work:

```bash
uv tool install examshield
```

## Quick start

After [Installation](#installation), run these in the repository folder:

```bash
uv run examshield protect exam.pdf --out out/copies --profile a-remap-decoy  # make a copy and check it right away
uv run examshield check exam.pdf out/copies/exam.pdf                         # check again that the content is the same
uv run examshield verify --profile a-remap-decoy --models local              # measure the effect of this profile
```

`protect` writes the copy (`exam.pdf`) and a job record (`exam.job.json`). Only the PDF is for distribution. The job record holds the answers the decoys point at, so do not hand it out. `--models local` uses text-layer extraction, Tesseract (if installed) and Ollama (if running). Whatever is missing is skipped, and the report says why.

Example `verify` output (2026-11-20, profile `a-remap-decoy`, on a machine with English Tesseract data only and no Ollama). The columns are model, path, group, legibility, accuracy, decoy hit rate and errors:

```
모델           경로           구분    글자 판독률  정답률  미끼 적중률  오류
text-layer     텍스트 레이어  대조군  100.0%       -       -            0
text-layer     텍스트 레이어  변환본  21.5%        -       -            0
tesseract-eng  이미지         대조군  40.4%        -       -            0
tesseract-eng  이미지         변환본  40.4%        -       -            0
건너뜀: ollama — Ollama(http://127.0.0.1:11434)에 연결하지 못했다. `ollama serve`가 떠 있는지 확인한다.
```

Extracting the text layer gets only 21.5% of the original right (대조군 is the control group, 변환본 the protected copy). The path that reads the page as a picture (Tesseract) scores the same for both, which means this profile does not stop an AI that reads the picture. The Tesseract number itself is low because the items are in Korean and the Korean data is missing.

`uv run examshield profile list` shows the profiles and their verification state. All six `a-` profiles are `experimental`: their effect has not been confirmed against a real AI.

## Usage

Every command is explained in **[USAGE.md](USAGE.md)** (in Korean). The original design of the commands is in [spec/06-commands.md](spec/06-commands.md).

### Develop this repository

```bash
uv sync --extra cloud    # the cloud SDKs are needed for the full test suite
uv run pytest            # tests
uv run ruff check .      # lint
uv run mypy              # type check
```

To turn on the checks before push, run this once after cloning:

```bash
git config core.hooksPath .githooks
```

After that, ruff, mypy and pytest run right before `git push`, and the push is blocked if any of them fails. No server-side CI is used. The check runs on your own machine, so `git push --no-verify` skips it.

## Known limitations

- The three scenario A techniques do not change what is shown on screen. They do nothing against a screenshot or an AI that reads the page as a picture. In the measurements, OCR legibility on the image path was the same before and after.
- Glyph mapping scrambling changes only the PDF's ToUnicode table. An extractor that reads the character table inside the font file can recover the original text. Only MuPDF and poppler (`pdftotext`) were checked.
- A hidden decoy points at a randomly chosen number, without knowing the right answer. It can point at the right answer by chance. Whether a real AI follows the decoy has not been measured.
- The rendering comparison in `check` uses MuPDF only. It cannot catch a case that looks different in another PDF viewer. A rasterized copy equals the original at its own resolution (200 dpi by default) and is blurrier than the original when zoomed further.
- Word (.docx) input was tested only with a stand-in converter on a machine without LibreOffice. Whether LibreOffice turns the same file into the same PDF every time, and whether it stays off the network while converting, was not confirmed.
- The tracking watermark is a faint background pattern. It survives screenshots, scaling, cropping and JPEG saving, but it is erased if brightness is raised until the background turns pure white. In that case `trace` names nobody. It also does not stop averaging several copies together, or a leak that retypes the text.
- `trace` needs the untransformed original, the mapping table and the seed file. Photos of paper (perspective distortion) are not handled. That is in the planned paid scope.
- Watermark recovery was checked with PDFs turned into pictures by MuPDF and poppler (72 to 220 dpi, partial crops, JPEG quality 60 and above). It was not checked with a real screen-capture tool or with pictures recompressed by a messenger.
- How visible the watermark pattern is to people was not measured. It darkens the background by at most 6/255 (about 2%). A faint grid may show on some monitors.
- For a watermarked copy, the tolerance of the rendering comparison in `check` widens from 2/255 to 8/255. In return, `check` verifies that the overlay on blank background has the shape of the watermark grid. Faint text laid over the page is caught; a change of 8/255 or less hidden on or at the edge of glyphs, and a picture drawn in whole pattern cells, are not.
- The cloud model clients have never been called with a real key. The requests each vendor's SDK sends were only received and checked by a fake server on the same machine. The fake server's responses were written from memory, so real responses may differ.
- Of the default model names (`gpt-6.2-sol`, `claude-opus-5-5`, `gemini-3.8-flash`), the OpenAI and Google ones are guesses picked from the names listed in the installed SDKs. Whether they are those vendors' flagship models was not confirmed. They can be changed with environment variables.
- `check` does not run OCR on a page whose rendering is pixel-identical to the original, because the OCR result would be identical too. For such pages this skips the step in spec section 6.2 ("OCR the rendering and compare"). `protect` on a 12-page licence PDF went from about 20 seconds to about 2. Pages whose rendering differs at all are still read by OCR, so a long document with many such pages is still slow.
- There are no scores from a real AI model yet. The Ollama client was tested only against a fake server that imitates the response format. If the real Ollama answers differently, or reports vision support differently, it will need fixing.
- Tesseract on the development machine has no Korean data, so the Korean OCR baseline is not a real value.
- The capture simulation has never been calibrated against real printing and photographing.
- Item set v1 is 12 easy general-knowledge items. Accuracy on it carries little statistical weight.
- `mock-reader` under `--models mock` is not a real AI. It exists to check the measuring pipeline.
- Step 0 of the spec (the feasibility experiment) was not done. It needs real AI models and API keys, and can be run with `verify`.
- Research has found that scrambled fonts, contrast distortion and interference lines do almost nothing against current multimodal AI. If no technique works, the direction narrows to "decoy + tracking" ([spec/12-risks.md](spec/12-risks.md)).
- No profile has a verification record. `verify` has never been run against a real AI model. The drift warnings were therefore tested only with hand-made reports.
- The drift bar (for each AI model, on the path it solved best, accuracy on the protected copy at most half of the control) and the 30 days are values chosen here; the spec gives no numbers. With 12 items, one or two answers can flip the verdict.
- Marking a profile that fell below the bar as `deprecated` is not automatic. `protect` warns that it should be changed, and a person edits the profile YAML.
- The safety hooks are a safety net that matches command text. A wrapped or indirect command can slip past them.

## Paid plans (planned)

Everything you run alone on your own machine is free. Features an institution needs to run exams (profiles for printed exams, photo forensics, an air-gapped installer, a commercial license, a subscription to verified profiles) are planned as paid plans. Nothing is for sale yet, and prices and details are not decided.

## Safety hooks

This repository ships Claude Code hooks (`.claude/settings.json`, `.claude/hooks/`). While you develop in this repo with Claude Code:

- **Blocked before running** (`block-dangerous.sh`): broad recursive deletes, git force pushes, `reset --hard`, `clean -fd`, staging secret files or `docs/` and `secrets/`, `curl | sh`, publishing the package.
- **Lint on every edit** (`lint-file.sh`): ruff runs on each edited Python file, and a failure is sent back to the agent.
- **Tests before finishing** (`run-tests.sh`): the agent cannot finish while tests fail.
- **Read-deny rules**: `.env`, `.env.*`, `secrets/`.

The hooks exit with code 2 on failure, because that is what sends the failure text back to the agent. The hooks have their own tests in `tests/test_hooks.py`.

## Data and privacy

- Seed files and the examinee mapping table belong in `secrets/`. `protect --copies` refuses to put either inside the output folder. That folder, `*.seed` and `.env` are excluded by `.gitignore`. The institution keeps them itself.
- The output folder (`out/`) and logs are not committed either.
- Logs hold only file hashes, the profile name and the time. They never hold question text.

## Project layout

```
Gotcha/
├── examshield/
│   ├── cli.py         # Typer entry point: protect, check, verify, report, auth, trace, profile
│   ├── protect/       # pipeline, copies, techniques/(rasterize, glyph_remap, decoy, watermark)
│   ├── check/         # compare: structure, text, rendering, OCR
│   ├── trace/         # detect: aligns a leak with the original and finds the examinee pattern
│   ├── benchmark/     # item set (v1) and PDF rendering
│   ├── profiles/      # profile YAML: none, none-print, a-*. verified/ holds the latest verification records
│   ├── verify/        # runner, capture_sim, metrics, report, drift, models/
│   ├── auth/          # keys: finds API keys in environment variables and the OS keychain
│   ├── providers.py   # cloud vendors: key names, default models, hosts
│   ├── netguard.py    # refuses network access when offline is set
│   └── pdfio.py       # PDF reading, rendering, deterministic saving
├── tests/
├── spec/              # the work spec (the original, split by section)
├── .claude/
│   ├── settings.json  # deny rules, hook registration
│   ├── hooks/         # blocking, lint and test hooks
│   ├── skills/        # phase-work: how to carry out one phase
│   └── agents/        # fresh-reviewer: a review with fresh eyes
├── .githooks/         # pre-push: ruff, mypy, pytest before a push
├── CLAUDE.md          # short guidance for AI agents
├── USAGE.md           # usage guide (Korean)
└── pyproject.toml
```

`docs/` is excluded from git and holds only notes on the paid plans.

## Roadmap

Each phase is reviewed and approved before the next one starts.

- [x] **Phase 0** — project skeleton: spec split, hooks and deny rules, lint, type check and checks before push, first draft of the docs
- [x] **Phase 1** — `verify`: item set, capture simulation, baselines (text extraction, Tesseract), local model, report. Done when: the untransformed control group gets a score
- [x] **Phase 2** — `protect` + `check`, scenario A profiles (rasterizing, glyph mapping scrambling, hidden decoys). Done when: output is deterministic and the zero-network tests pass
- [x] **Phase 3** — `auth`, cloud model verification, `--offline`. Done when: running without a key explains what to do and exits, and the confirmation step before a call works
- [x] **Phase 4** — tracking watermark, `--copies`, `trace`. Done when: the examinee number is recovered from a PDF and from a screenshot-quality image
- [x] **Phase 5** — model drift warnings, `report`, `profile`, finishing the docs. Done when: all tests pass
- [ ] **Phase 6** — scenario B print profiles, calibration against real printing and photographing, watermark recovery from photos (planned, paid)
- [ ] **Phase 7** — single executable for air-gapped networks, commercial license, profile subscription, web UI, HWP input (planned, paid)

How this maps to the plan in the spec ([spec/11-plan.md](spec/11-plan.md)): spec steps 1 to 4 are Phases 1 to 4, spec step 5 is Phase 6, and spec step 6 is Phase 7. `examshield trace` is not in the spec's command list; it is added because watermark recovery needs it.

### Tech stack

| Area | Choice |
|---|---|
| Language, package management | Python 3.12, uv |
| CLI | Typer |
| PDF reading, writing, rendering | PyMuPDF, with pikepdf as a helper |
| Fonts | fontTools |
| Images, capture simulation | OpenCV, Pillow, NumPy |
| OCR baseline | Tesseract |
| Local models | Ollama |
| Cloud models | each vendor's official Python SDK (optional install `examshield[cloud]`) |
| Key storage | keyring |
| Quality | pytest, hypothesis, ruff, mypy strict, git pre-push hook |

## Contributing

The project is at an early stage. Please open an issue to discuss a change before sending one.

## License

Not decided. PyMuPDF is AGPL, so the choice has to fit with that ([spec/13-open-decisions.md](spec/13-open-decisions.md)). Until it is decided, no redistribution terms are granted.
