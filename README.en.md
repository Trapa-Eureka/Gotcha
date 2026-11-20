# Gotcha — examshield

An offline CLI that reworks an exam document on your machine (PDF, Word, Hangul HWP) while keeping its content the same, so that copying its text into an AI is blocked and a leaked copy can be traced to whose it was, and shows the effect with numbers stamped with a date and model versions.

> **Status: early development (implemented through Phase 7; checks against real AI, real printing and photographing, and an air-gapped PC are still open).**
> `protect` (makes the copies), `check` (checks the content is the same), `verify` and `report` (measure the effect) `trace` (finds whose copy leaked) `profile` (lists profiles and their verification state) `calibrate` (measures the capture conditions of a photo of a printed page) and `ui` (a web page that opens on this machine only) work. Only one real AI model has been measured: Claude (`claude-opus-5-5`), on two profiles (2026-11-21). The result: they block only the path where text is copied and pasted, not screenshots or file uploads. OpenAI and Google models have never been called.
> Phases 1 to 5 and Phase 6 (printed exams) are built. Of the two measured profiles, `a-remap-decoy` blocked only the text-copy path and `a-decoy` blocked no path; the rest have never been measured against a real AI, so `protect` prints a warning for every `a-` and `b-` profile. The printed-exam features were checked with the simulation only; nothing was actually printed and photographed.

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
| `examshield verify` — measures a profile on the built-in item set and writes a report | Done (control group, scenario A and B profiles) |
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
| Scenario B (printed exams): dot-pattern profile `b-dots`, examinee number recovery from a photo of the printed page, `examshield calibrate` | Implemented (checked with the capture simulation only, never with real printing and photographing; per-item semantic decoys are not built) |
| Single executable for air-gapped networks (macOS) | Implemented (run on the Mac that built it; not checked on another Mac or on a machine without network; no Windows build yet) |
| Local web page `examshield ui` — make a copy, check sameness, list profiles | Implemented (listens on 127.0.0.1 only) |
| Hangul (.hwp, .hwpx) input | Implemented (tested only with a stand-in converter, without LibreOffice; never checked with a real Hangul file) |
| License | AGPL-3.0-or-later applied. The profile subscription will not be a paid product |

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
- Optional: [Tesseract](https://github.com/tesseract-ocr/tesseract) with `kor` and `eng` data (OCR baseline), [Ollama](https://ollama.com) with a vision model (local model measurement), LibreOffice (Word and Hangul input)

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

For a machine without Python and uv, build a single executable. This works on macOS only.

```bash
uv sync
packaging/build-macos.sh    # dist/examshield (about 120 MB)
```

Copy the file to another Mac and run it as is. Tesseract and LibreOffice are not inside. If that machine has them installed they are found and used; without them PDF processing still works.

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
- A hidden decoy points at a randomly chosen number, without knowing the right answer. It can point at the right answer by chance (2 to 5 of 12 items in the measurements). `protect --answer-key` takes an answer key and avoids the right answers, but then the hidden decoy text reveals one number that is not the answer. Measured with Claude as if a key had been given, accuracy on the text-copy path fell from 41.7% to 0% (it followed the decoy on all 12 items, 2026-11-21). When Claude could read the question it mostly ignored the decoy (2 of 10 items on the text path, 0 on the picture and PDF paths). It followed the decoy only when the text was scrambled so the question could not be read (7 of 7 items). With the decoy sentence drawn plainly above the question and the page given as a picture, it still did not follow it (0 of 8 items, accuracy 100%). So visible decoys and faintly hidden decoys were not built. This was measured with one model and one wording.
- The rendering comparison in `check` uses MuPDF only. It cannot catch a case that looks different in another PDF viewer. A rasterized copy equals the original at its own resolution (200 dpi by default) and is blurrier than the original when zoomed further.
- Word (.docx) and Hangul (.hwp, .hwpx) input were tested only with a stand-in converter on a machine without LibreOffice. Whether LibreOffice turns the same file into the same PDF every time, and whether it stays off the network while converting, was not confirmed.
- The on-screen tracking watermark (`a-watermark`, `a-full`) is a faint background pattern. It survives screenshots, scaling, cropping and JPEG saving, but it is erased if brightness is raised until the background turns pure white. In that case `trace` names nobody. It also does not stop averaging several copies together, or a leak that retypes the text.
- `trace` needs the untransformed original, the mapping table and the seed file. Photos of paper are handled only for dot-pattern (`b-dots`) copies. The on-screen watermark disappears in print.
- Watermark recovery was checked with PDFs turned into pictures by MuPDF and poppler (72 to 220 dpi, partial crops, JPEG quality 60 and above). It was not checked with a real screen-capture tool or with pictures recompressed by a messenger.
- How visible the watermark pattern is to people was not measured. It darkens the background by at most 6/255 (about 2%). A faint grid may show on some monitors.
- For a watermarked copy, the tolerance of the rendering comparison in `check` widens from 2/255 to 8/255. In return, `check` verifies that the overlay on blank background has the shape of the watermark grid. Faint text laid over the page is caught; a change of 8/255 or less hidden on or at the edge of glyphs, and a picture drawn in whole pattern cells, are not.
- Of the cloud model clients, OpenAI and Google have never been called with a real key. Only Anthropic was actually called. The requests each vendor's SDK sends were only received and checked by a fake server on the same machine. The fake server's responses were written from memory, so real responses may differ.
- Of the default model names (`gpt-6.2-sol`, `claude-opus-5-5`, `gemini-3.8-flash`), the OpenAI and Google ones are guesses picked from the names listed in the installed SDKs. Whether they are those vendors' flagship models was not confirmed. They can be changed with environment variables.
- `check` does not run OCR on a page whose rendering is pixel-identical to the original, because the OCR result would be identical too. For such pages this skips the step in spec section 6.2 ("OCR the rendering and compare"). `protect` on a 12-page licence PDF went from about 20 seconds to about 2. Pages whose rendering differs at all are still read by OCR, so a long document with many such pages is still slow.
- `--copies` slows down in proportion to the number of copies. A 12-page licence PDF with 5 examinee numbers took about 7 seconds (both `a-full` and `a-watermark`), about 1.1 seconds per copy. Most of it is the watermark check run on every copy (verifying cell by cell that the overlay has the grid shape). At that rate a few hundred examinee numbers take from a few minutes to over ten. It has not been run at that scale, and the time `trace` needs to compare a few hundred numbers in the mapping table has not been measured either.
- The printed-exam features (`b-dots`, recovery from photos, `calibrate`) were checked with the capture simulation only. They have never been tried with a real printer and camera. One dot is 0.5 pt (about 0.18 mm) on a side, so depending on printer resolution, toner spread and paper it may not print or may smear. Recovery from photos of curled or folded paper, of only part of a page, or of a photocopy was not checked either.
- The dot pattern does not stop an AI from reading the page. Given pictures that imitate printing and photographing, Claude scored 100% legibility and accuracy, the same as without the pattern (2026-11-21, one model, 12 items, simulation). `b-dots` is therefore treated as a leak-tracing profile only. The dots are visible (about 0.35% of the page area), and their effect on human legibility was not measured. Examinees with low vision need the original without dots, and that copy cannot be traced.
- For a dotted copy, `check` verifies that the page with the pattern lifted off equals the original, and that the pattern consists only of black dots at cell centres. Large letters or pictures drawn with cell-centre dots alone cannot be told apart from the pattern.
- The per-item semantic decoys the spec lists for scenario B are not built. A person has to design them item by item, so they cannot be added automatically.
- The capture conditions `calibrate` prints are estimates. Even on simulated captures, noise and blur read lower than the true values. That is why it shows the real photo and the simulation side by side, measured the same way.
- The only scores from a real AI are one Claude model, 12 items and two profiles. Other vendors' models, local models and the other profiles were not measured. The Ollama client was tested only against a fake server that imitates the response format. If the real Ollama answers differently, or reports vision support differently, it will need fixing.
- Tesseract on the development machine has no Korean data, so the Korean OCR baseline is not a real value.
- The capture simulation has never been calibrated against real printing and photographing.
- Item set v1 is 12 easy general-knowledge items. Accuracy on it carries little statistical weight.
- `mock-reader` under `--models mock` is not a real AI. It exists to check the measuring pipeline.
- Step 0 of the spec (the feasibility experiment) was not done. It needs real AI models and API keys, and can be run with `verify`.
- Research has found that scrambled fonts, contrast distortion and interference lines do almost nothing against current multimodal AI. If no technique works, the direction narrows to "decoy + tracking" ([spec/12-risks.md](spec/12-risks.md)).
- There are only two verification records, for `a-remap-decoy` and `a-decoy`. Scenario A techniques do not change what is shown on screen, so the screenshot and file-upload paths cannot be blocked for any model. So the verdict is now given per path. `a-remap-decoy` does block the text-copy path and reads "blocks some paths only"; `a-decoy` fails even that path and is marked `deprecated`. `deprecated` is used only when a profile fails a path it claims to block. This is narrower than spec section 7 ("mark profiles below the bar as `deprecated`").
- The drift bar (for each AI model, on the path it solved best, accuracy on the protected copy at most half of the control) and the 30 days are values chosen here; the spec gives no numbers. With 12 items, one or two answers can flip the verdict.
- Marking a profile that fell below the bar as `deprecated` is not automatic. `protect` warns that it should be changed, and a person edits the profile YAML.
- For a Hangul document, the PDF that LibreOffice produces is taken as the original. LibreOffice may not carry over fonts, tables and page breaks faithfully, and some versions cannot open the file at all. `check` guarantees only that the converted PDF and the copy are the same. Whether it looks like what Hangul shows has to be checked by a person. The safe route is a PDF exported from Hangul itself.
- The single executable was run only on the Mac that built it (Apple Silicon). It was not checked on another Mac, an Intel Mac or a machine without network. It is not signed or notarized, so Gatekeeper may block it on another Mac. It unpacks into the system temp folder on every start and takes about 4 seconds to start. There is no Windows build yet.
- Tesseract and LibreOffice are not in the executable. LibreOffice is several hundred MB, and bundling Tesseract means relocating Homebrew libraries, which breaks easily. Without them the OCR comparison is skipped (and that is stated), and Word and Hangul input are refused. The cloud SDKs are left out too, so the executable cannot verify against cloud models.
- The web page only makes a copy (one file), checks sameness and lists profiles. Per-examinee copies, tracing and measuring are done on the command line. Only Chrome was used to confirm the page opens. Uploads are handled in memory (up to 100 MB per request), one document at a time.
- The safety hooks are a safety net that matches command text. A wrapped or indirect command can slip past them.

## Pricing

Nothing is for sale. The code for every feature is public in this repository under the AGPL. The profile subscription will not be a paid product. Profiles and their verification records live in the repository.

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
│   ├── cli.py         # Typer entry point: protect, check, verify, report, auth, trace, profile, calibrate
│   ├── protect/       # pipeline, copies, techniques/(rasterize, glyph_remap, decoy, watermark, dots)
│   ├── check/         # compare: structure, text, rendering, OCR
│   ├── trace/         # detect: aligns a leak with the original and finds the examinee pattern
│   ├── benchmark/     # item set (v1) and PDF rendering
│   ├── profiles/      # profile YAML: none, none-print, a-*, b-*. verified/ holds the latest verification records
│   ├── verify/        # runner, capture_sim, calibrate, metrics, report, drift, models/
│   ├── auth/          # keys: finds API keys in environment variables and the OS keychain
│   ├── ui/            # server, page.html: the web page that opens on this machine only
│   ├── convert.py     # turns Word and Hangul files into PDF with LibreOffice
│   ├── providers.py   # cloud vendors: key names, default models, hosts
│   ├── netguard.py    # refuses network access when offline is set
│   └── pdfio.py       # PDF reading, rendering, deterministic saving
├── tests/
├── packaging/         # build-macos.sh: builds the single executable
├── spec/              # the work spec (the original, split by section)
├── .claude/
│   ├── settings.json  # deny rules, hook registration
│   ├── hooks/         # blocking, lint and test hooks
│   ├── skills/        # phase-work: how to carry out one phase
│   └── agents/        # fresh-reviewer: a review with fresh eyes
├── .githooks/         # pre-push: ruff, mypy, pytest before a push
├── CLAUDE.md          # short guidance for AI agents
├── USAGE.md           # usage guide (Korean)
├── LICENSE            # AGPL-3.0
└── pyproject.toml
```

`docs/` is excluded from git and holds only business notes.

## Roadmap

Each phase is reviewed and approved before the next one starts.

- [x] **Phase 0** — project skeleton: spec split, hooks and deny rules, lint, type check and checks before push, first draft of the docs
- [x] **Phase 1** — `verify`: item set, capture simulation, baselines (text extraction, Tesseract), local model, report. Done when: the untransformed control group gets a score
- [x] **Phase 2** — `protect` + `check`, scenario A profiles (rasterizing, glyph mapping scrambling, hidden decoys). Done when: output is deterministic and the zero-network tests pass
- [x] **Phase 3** — `auth`, cloud model verification, `--offline`. Done when: running without a key explains what to do and exits, and the confirmation step before a call works
- [x] **Phase 4** — tracking watermark, `--copies`, `trace`. Done when: the examinee number is recovered from a PDF and from a screenshot-quality image
- [x] **Phase 5** — model drift warnings, `report`, `profile`, finishing the docs. Done when: all tests pass
- [ ] **Phase 6** — scenario B print profiles, calibration against real printing and photographing, watermark recovery from photos. The code is implemented. The completion criterion, "compare the simulation with real photos", needs pages that were actually printed and photographed, so it is still open
- [ ] **Phase 7** — single executable for air-gapped networks, license, web UI, Hangul input. The code is implemented (the executable for macOS only). The completion criterion, "install and run on an air-gapped PC", has to be tried on such a PC, so it is still open, as is a Windows executable. The profile subscription will not be a paid product, so it is not built

How this maps to the plan in the spec ([spec/11-plan.md](spec/11-plan.md)): spec steps 1 to 4 are Phases 1 to 4, spec step 5 is Phase 6, and spec step 6 is Phase 7. `examshield trace` and `examshield calibrate` are not in the spec's command list; they are added because watermark recovery and the comparison with real photos need them.

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
| Web page | standard library `http.server`, one HTML page that loads nothing from outside |
| Single executable | PyInstaller |
| Quality | pytest, hypothesis, ruff, mypy strict, git pre-push hook |

## Contributing

The project is at an early stage. Please open an issue to discuss a change before sending one.

## License

[GNU Affero General Public License v3.0 or later](LICENSE) (AGPL-3.0-or-later). PyMuPDF, which does the PDF work, is AGPL, so distributing this tool means following the AGPL anyway; the same license was chosen to match. If you distribute a modified version or offer it as a network service, you must publish your modified source under the same terms.

Using it without publishing source needs separate permission from the copyright holder of this repository, and a commercial PyMuPDF license bought from Artifex. No such commercial license is on sale. This paragraph is not legal advice.
