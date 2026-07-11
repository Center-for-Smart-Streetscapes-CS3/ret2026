# CS3 Research Experience for Teachers

This Jekyll site contains course materials for the 2024, 2025, and 2026 CS3 RET cohorts. It uses the design files in `unstructured/` and the GitHub Pages setup from [RET 2024](https://github.com/Center-for-Smart-Streetscapes-CS3/ret2024). The site opens to 2026 by default.

## What is included

- Home, About, Calendar, Schedule, and Staff routes
- Complete 2024, 2025, and 2026 course calendars with cohort and resource-type filters
- Responsive navigation and expandable weekly sections
- Current staff profiles and portraits, updated from [RET 2025](https://github.com/Center-for-Smart-Streetscapes-CS3/ret2025)
- WebP images totaling less than 2 MB
- GitHub Pages deployment workflow

## Run locally

Prerequisites: Ruby 3 and Bundler.

```sh
bundle install
bundle exec jekyll serve --livereload
```

Open `http://127.0.0.1:4000/ret2026/`.

To run at the domain root while editing:

```sh
bundle exec jekyll serve --livereload --baseurl ""
```

## Update content

- Calendar entries: `_data/calendar.yml`
- Staff profiles: `_staffers/`
- Shared page chrome: `_layouts/default.html`
- Page content: `index.html`, `about.html`, `calendar.html`, `schedule.html`, and `staff.html`
- Styles and interactions: `assets/css/` and `assets/js/`

Items without a URL in `_data/calendar.yml` are shown as plain text.

## Deploy to GitHub Pages

1. In the repository, open **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Push to `main`, or manually run **Deploy Jekyll site to Pages** from the Actions tab.

The workflow builds Jekyll and publishes `_site` to the `github-pages` environment. The expected project URL is:

`https://center-for-smart-streetscapes-cs3.github.io/ret2026/`

If the repository is renamed, update `baseurl` in `_config.yml` to `/<repository-name>`.

## Source prototype

`unstructured/` contains the local design export and is excluded from deployment. The maintained site files are at the repository root.
