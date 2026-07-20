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

## The learning path

Progress flows through four level sets on the **Quest Map**, Duolingo style:
completing a mission returns the student to the map (which fills the work
area), where the next card glows with a "Next" badge and one click starts it.
Every mission opens with a large purpose box built from its `title`, `goal`,
`story` (Foundations), and chips (dataset, goal, focus) before play begins.

1. **Foundations of Learning** (`content/foundations.json`) — four worked
   examples on a dedicated screen (nav tab 00): a real problem becomes a
   feature row, the row meets a weight matrix and a bias, nonlinearity is
   shown to be necessary, and weights are fitted by hand. The network graph
   and the matrix equation are hover linked: pointing at an edge, node, or
   cell lights up its partners. New students land here.
2. **Training Camp** (`content/basics-missions.json`) — five intro missions
   that reveal the Playground interface one element at a time: the data panel,
   weights, probabilities and the margin, activation functions, the loss curve.
3. **Model Playground** (`content/missions.json`) — eight missions, unlocked
   when Training Camp is complete.
4. **Text Lab** (`content/text-lessons.json`) — eight lessons on the same map.

Student progress (XP, per-set completion, current position) is stored in
`localStorage` under `cs3_toolkit_v2`. Sandbox mode ignores all gating and
shows every control.

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

### Missions — `basics-missions.json` and `missions.json`

Each file is an array; order in the file = mission order, and missions unlock
sequentially. Add, remove, or reorder freely — all counters adapt.

```jsonc
{
  "title":       "Separate the Two Clouds",
  "dataset":     "gaussian",           // gaussian | xor | circle | spiral
  "goal":        "Shown under the title.",
  "hint":        "Shown when the student presses Hint.",
  "restrict":    "Short label shown on the quest map card.",
  "target":      0.08,                 // validation loss to beat; null = no loss requirement
  "targetLabel": "explore the data",   // GOAL chip text when target is null
  "hidden":      [3],                  // starting hidden layers; [] = no hidden layer
  "noise":       30,                   // optional, % label noise preset
  "question":    "Classroom discussion question (reference for teachers).",
  "answer":      "Shown on the quest map after the mission is completed.",
  "show":        ["boundary", "transport"],  // optional, see "Progressive UI" below
  "rules":       [ ... ]               // extra pass/fail requirements, see below
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
| `activations-min` | student has tried ≥ `value` different activation functions |
| `epochs-min` | the model has trained ≥ `value` epochs |
| `regen-min` | student pressed Regenerate examples ≥ `value` times |
| `inspect-edge` | student clicked a connection to inspect its weight |
| `discrete-toggled` | student toggled the Discrete checkbox |
| `discrete-off` | the Discrete checkbox is off |
| `any` / `all` | nested `rules` array: at least one / all must pass (shares one `msg`) |

Feature keys: `x1, x2, x1sq, x2sq, x1x2, sinx1, sinx2`. The interaction
counters (`regen-min`, `inspect-edge`, ...) reset each time a mission opens.

### Progressive UI — the `show` field

A mission that carries a `"show"` list displays only those interface groups;
everything else stays hidden until a later mission (or Sandbox mode) reveals
it. Missions without `show` display everything. Group names:

`boundary` (kept in every list), `transport` (play / step / reset), `metrics`
(epoch and loss bar), `inspector` (selected unit panel), `boundaryopts` (Test
and Discrete checkboxes), `tab-model`, `tab-train` (control tabs),
`layercontrols` (the − / + pills on the network), `loss` (loss chart),
`views` (activations / weights / gradients), `presets` (Shallow / Wide /
Deep), `features` (input feature chips).

### Foundations — `foundations.json`

An array of worked examples, each with `title`, `restrict` (map card label),
`story` (the real problem), `goal`, `hint`, `note` (data provenance, e.g.
"Illustrative numbers"), `question`, `answer`, a `type`, and a `check`:

| type | shows | numbers it needs |
|---|---|---|
| `linear` | 1×2 row · 2×2 matrix + 1×2 bias, expandable outputs | `features` (label+value), `outputs` (labels), `W`, `b` |
| `argmax` | same, plus a winner banner and input switcher | as above, plus `examples` (`label`, `x`) |
| `compose` | two stacked layers next to the collapsed single layer, with a ReLU toggle | `x`, `W1`, `b1`, `W2`, `b2` |
| `fit` | +/− steppers on w₁, w₂, bias over a table of labeled examples | `features` (labels), `outputLabel`, `examples` (`x`, `y`), `start` |

`check` fields (each with its message): `minCells`/`cellsMsg` (distinct
weights hovered), `minInputs`/`inputsMsg` (examples tried), `requireToggle`/
`toggleMsg` (ReLU switched on), `maxError`/`errorMsg` (mean absolute error at
or below the value; `{err}` in the message is filled at runtime).

### Text Lab lessons — `text-lessons.json`

An array, same ordering behavior. Each lesson presets the lab
(`tokenizer`: char|word|syllable|bpe, `model`: markov|rnn|gpt, `order`: 1|2,
`source`: nursery|tempest|shakespeare-all|tinystories) and has `restrict`
(quest map card label), `goal`, `hint`, and `notice` (shown on the quest map
after completing). The `check` object defines what *Check Lesson* requires —
every field is optional, each has a matching message field:

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
  `data-tour="..."` attribute in the HTML; steps whose target is hidden at the
  current UI stage are skipped automatically.
- **`corpus-samples.json`** — the short built-in corpora (`nursery`,
  `tempest` fallback, `analogy`). Big corpora live in `data/*.js`.
- **`ui-text.json`** — level set titles and descriptions (`sets`), dataset
  display names, per-model notice lines, sandbox card copy, and shared
  messages. Messages may contain `{placeholders}` — keep them, they are
  filled in at runtime.
- **`config.json`** — XP tuning: `xpPerLevel`, `foundationsXP`, `basicsXP`,
  `missionXP`, `lessonXP`, and the small awards for stepping/generating.

## How content is wired in (for developers)

The component in `index.html` fetches all eight JSON files in `loadContent()`
before initializing (`initApp()`), assigning them to `this.foundations`,
`this.basics`, `this.missions`, `this.textLessons`, `this.tour`,
`this.corpusSamples`, `this.ui`, and `this.cfg`. Mission rules are evaluated
by `ruleOK()`; lesson checks by `lessonOK()`; Foundations rendering and its
hover linking by the `f`-prefixed methods (`fRenderStage()`, `fHover()`,
`fCheckOK()`); UI disclosure by `applyUIStage()` from each basics mission's
`show` list; the quest map and completion flow live in `renderQuest()` /
`openQuest()` / `nextTarget()`.
