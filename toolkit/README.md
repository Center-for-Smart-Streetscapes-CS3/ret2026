# CS3 AI Training Toolkit

A local, no-network teaching app for neural networks and small language models.
Two labs: the **Model Playground** (networks & decision boundaries) and the
**Text Lab** (tokens, Markov chains, RNNs, Mini GPT).

## Running it

Double-click **`Start AI Toolkit.cmd`**. It starts a tiny local server
(`tools/serve.ps1`, no installs needed) and opens the app in your browser.

> Why a server? All lesson text lives in editable `content/*.json` files, and
> browsers refuse to read local JSON from a page opened via double-click
> (`file://`). Served from `http://localhost` everything works.

The folder is fully static: upload it to any static host (GitHub Pages, S3,
nginx, `python -m http.server`, ...) and the site serves from `index.html`
with no build step.

## Folder map

| Path | What it is | Edit? |
|---|---|---|
| `index.html` | The app: page layout + all behavior (one component class in a `<script>` at the bottom) | code changes only |
| `content/` | **All teaching content — missions, lessons, hints, tour, XP** | **yes — this is the folder you edit** |
| `data/` | Large training corpora (Shakespeare, TinyStories) as `.js` globals | rarely |
| `minigpt-engine.js` | The Mini GPT / RNN training engine (`window.MiniGPT`) | code changes only |
| `support.js` | Generated framework runtime — **do not edit** | never |
| `tools/serve.ps1` | The local server behind `Start AI Toolkit.cmd` | rarely |
| `uploads/` | Raw source texts kept for reference | no |

After editing any `content/*.json`, just **reload the browser tab** — no build
step. If the app shows a "Could not load the toolkit content" screen, the named
file has a JSON syntax error (a trailing comma is the usual culprit).

## Editing the content

### `content/missions.json` — Playground missions

An array; order in the file = mission order, and missions unlock sequentially.
Add, remove, or reorder freely — all "Mission X of Y" counters adapt.

```jsonc
{
  "title":    "Separate the Two Clouds",
  "dataset":  "gaussian",            // gaussian | xor | circle | spiral
  "goal":     "Shown under the title.",
  "hint":     "Shown when the student presses Hint.",
  "restrict": "Short label shown on the quest map.",
  "target":   0.08,                  // validation loss the student must beat
  "hidden":   [3],                   // starting hidden layers, e.g. [7,5]
  "noise":    30,                    // optional, % label noise preset
  "question": "Classroom discussion question (reference for teachers).",
  "answer":   "Shown as guidance after the mission is completed.",
  "rules":    [ ... ]                // extra pass/fail requirements, see below
}
```

**Rules** are checked in order when the student presses *Check Solution*; the
first failing rule's `msg` is shown. Available `type`s:

| type | passes when |
|---|---|
| `layers-exact` / `layers-min` / `layers-max` | number of hidden layers =, ≥, ≤ `value` |
| `neurons-min` | total hidden neurons ≥ `value` |
| `width-min` / `width-max` | widest hidden layer ≥ / ≤ `value` |
| `features-only` | selected input features are exactly `value` (e.g. `["x1","x2"]`) |
| `feature-any` | at least one of `value` is selected |
| `noise-min` | noise slider ≥ `value` |
| `optimizers-min` | student has tried ≥ `value` different optimizers |
| `discrete-off` | the "Discrete" checkbox is off |
| `any` / `all` | nested `rules` array: at least one / all must pass (shares one `msg`) |

Feature keys: `x1, x2, x1sq, x2sq, x1x2, sinx1, sinx2`.

### `content/text-lessons.json` — Text Lab lessons

An array, same ordering behavior. Each lesson presets the lab
(`tokenizer`: char|word|syllable|bpe, `model`: markov|rnn|gpt, `order`: 1|2,
`source`: nursery|tempest|shakespeare-all|tinystories) and has `goal`, `hint`,
and `notice` (shown after completing). The `check` object defines what *Check
Lesson* requires — every field is optional, each has a matching message field:

```jsonc
"check": {
  "model": "rnn",            "modelMsg":     "Requirement: ...",
  "order": 2,                "orderMsg":     "Requirement: ...",
  "tokenizer": "word",       "tokenizerMsg": "Requirement: ...",
  "requireSelection": true,  "selectionMsg": "Requirement: ...",  // clicked a token
  "minSteps": 20,            "stepsMsg":     "Requirement: ...",  // training updates
  "requireSample": true,     "sampleMsg":    "Requirement: ...",  // generated text
  "requireAnalogy": true,    "analogyMsg":   "Requirement: ..."   // solved an analogy
}
```

### The rest of `content/`

- **`tour.json`** — the first-run coach-mark tour. `sel` must match a
  `data-tour="..."` attribute in the HTML; `title`/`body` are free text.
- **`corpus-samples.json`** — the short built-in corpora (`nursery`,
  `tempest` fallback, `analogy`). Big corpora live in `data/*.js`.
- **`ui-text.json`** — dataset display names, per-model notice lines, sandbox
  card copy, and shared messages. Messages may contain `{placeholders}` —
  keep them, they are filled in at runtime.
- **`config.json`** — XP tuning: `xpPerLevel`, `missionXP`, `lessonXP`, and the
  small awards for stepping/generating.

## How content is wired in (for developers)

The component in `index.html` fetches all six JSON files in
`loadContent()` before initializing (`initApp()`), assigning them to
`this.missions`, `this.textLessons`, `this.tour`, `this.corpusSamples`,
`this.ui`, and `this.cfg`. Mission rules are evaluated by `ruleOK()`; lesson
checks by `lessonOK()`. Student progress (XP, completed missions, theme) is
stored in `localStorage` under `cs3_toolkit_v2`.
