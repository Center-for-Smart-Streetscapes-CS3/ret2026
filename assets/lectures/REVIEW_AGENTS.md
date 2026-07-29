# Review agents

Instructions for spawning independent subagents that review and fix the decks in this folder. Run them after every substantive revision, before delivery, at least the first two in parallel. Each agent reads GUIDELINES.md first; findings must quote the offending content exactly and name the violated rule, so the main agent can verify and apply fixes. Reviewers report; the main agent (or a dedicated fixer agent) edits.

## Agent 1: language reviewer

Prompt template:

> Read GUIDELINES.md in full; the "Language constructs" section is your rubric. Then extract the visible prose from <deck>.html: strip `<style>`, `<script>`, and tags; keep slide text, svg text labels, figcaptions, and data-title values. Judge every sentence and every slide title against each rule, with special attention to: dashes and hyphenated compounds; "X, not Y" and other contrastive rhetoric; trailing qualifier ambiguity; paratactic chains that hide causation; banned title shapes ("An X that Y", comma apposition tails); paired clause headers ("An X, and Y", as in "A tempting rule, and where it fails") anywhere, including misconception tags and captions; any subtitle on the title slide (none is allowed); agentless abstraction (an abstract noun acting with no mechanism, e.g. "the past arrives through every handoff"); dodged technical names (a function must be called a function); crutch word overuse (count content word frequencies across the deck and flag any conspicuous tic; "rule" is the documented offender); clipped shorthand (a verb missing its object, as in "the model missed by 1", which must name what was missed: "missed the right answer by 1"; a casual noun where the precise one exists, size for magnitude; a coined house noun such as "a miss" where mistake or error is the plain word); repeated phrase frames (any distinctive wording or sentence frame appearing more than once in the deck, or reused from another deck of the series, e.g. "An appealing, but X, Y"; one appearance per series is the limit); self narrating scaffolds in BOTH directions (forward: any lead-in that describes the prose itself, its structure, or the delivery plan, e.g. "The two accountings, each in one sentence:", "For scale, each with its source:", "Read aloud:"; backward: any sentence whose subject is the slide's own claims, sections, or lists as a collection, e.g. "Both statements are true at once:", "Both crafts are one craft:"; apply the test of replacing the subject with "what this slide just said", and flag the sentence if it still parses; production narration is the same defect: any mention of the deck, a slide, the build, or the authorship as subject matter, e.g. "when the deck was built", "drawn for the deck", "on this slide", "a later slide"; "(illustrative)" labels, sources slide provenance, and operational instructions the audience must act on stay allowed; a short label naming the subject itself, such as "History:", stays allowed); unsupported or unlabeled numbers; card grid usage in markup. Also flag any sentence a STEM high school teacher would need to read twice. Return a numbered list: {slide number and title, exact offending text, rule violated, suggested rewrite that obeys all rules}. Do not edit any file.

## Agent 2: geometry and visual reviewer

Prompt template:

> Review the rendered geometry of <deck>.html. First run the built in harness and read the JSON report:
> ```powershell
> cmd /c '"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless=new --disable-gpu --no-first-run --user-data-dir=%TEMP%\qaprof --virtual-time-budget=8000 --dump-dom "file:///<deck>.html?qa=1&frag=off&theme=dark" > %TEMP%\qa_dump.html 2>nul'
> ```
> then extract the `<pre id="qa-report">` JSON from the dump. Every entry (overflow, zoom below 0.92, text or block collisions, clipped svg content, arrow defects, tex-error entries from LaTeX that failed to render) is a finding. Second, screenshot EVERY slide in the dark theme and the diagram slides in the light theme (`--screenshot`, `--window-size=1440,900`, URL `?frag=off&theme=<t>#s<n>`), look at each image, and report: overlapping elements, arrows whose heads swallow their shafts or that pierce or miss boxes, labels sitting on lines, crowded or clipped figures, anything a careful human presenter would fix. Known artifact: headless Chromium occasionally paints one frame with the svg text layer displaced while shapes stay put; re-shoot a slide before reporting it, and only report defects that reproduce twice. Return findings as {slide, defect, evidence (report entry or screenshot name), suggested geometric fix with concrete coordinates or attributes}. Do not edit any file.

## Agent 3: pedagogy and facts reviewer

Prompt template:

> Read GUIDELINES.md (reveal pedagogy, concrete before abstract, real history) and the audience rules in ~/.claude/skills/manim/audiences/high-school.md. Extract the slide text of <deck>.html in slide order. Check: does every mechanism get a concrete worked example, with numbers a student could recompute, before the general rule; do relation slides start from the correct complete example before abstracting; are question frames present and answerable; is every historical claim (names, dates, quotes, counts) accurate against sources you verify on the web; is every invented number labeled illustrative. Return findings as {slide, issue, evidence or source, suggested fix}. Do not edit any file.

## Applying findings

1. Deduplicate and verify each finding against the file; discard anything that does not reproduce.
2. Apply fixes with the smallest edits that satisfy the rule; geometric fixes prefer the analytic systems (data-connect, data-connect-curve, auto-fit) over hand tuned coordinates.
3. Re-run the QA harness and the style audit; re-screenshot changed slides in both themes.
4. Record any NEW class of defect as a rule in GUIDELINES.md and in memory, so the next revision cannot repeat it.
