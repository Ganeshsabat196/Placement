# Data Scientist Interview Prep

A responsive study website for a two-round data science interview:

- **Preparation Plan**: a 3-week schedule and what interviewers look for
- **Concept Notes** (Round 1): statistics, ML, deep learning, NLP/GenAI, SQL, Python, MLOps
- **200 Interview Questions**: answers stay hidden until you open them; mark each one *known* or *revise*
- **29 Case Studies** (Round 2): frameworks and worked solutions, with a 25-minute practice timer

Site features: search across all pages (press `/`), progress tracking saved in your browser, filters by difficulty and status, a random question or case button, dark mode, and a layout that works on phones.

## Project structure

```
index.html                       ← the website
assets/style.css, assets/app.js  ← styling and behaviour
assets/content.js                ← generated from the .md files (don't edit by hand)
build.py                         ← regenerates assets/content.js
00_…md – 03_…md                  ← the content (edit these)
.nojekyll                        ← tells GitHub Pages to serve files as-is
```

## View it locally

Double-click `index.html`, or run a local server:

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

An internet connection is needed for the Markdown renderer, code highlighting and fonts, which load from CDNs.

## Edit the content

1. Edit any of the `.md` files.
2. Run `python3 build.py`.
3. Commit both the `.md` file and `assets/content.js`.

## Deploy on GitHub Pages

```bash
cd data_science_interview_prep
git init
git add .
git commit -m "Data science interview prep site"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Then on GitHub, go to **Settings → Pages → Build and deployment**. Set **Source** to *Deploy from a branch*, choose branch `main` and folder `/ (root)`, and save.
After a minute the site is live at `https://<your-username>.github.io/<repo-name>/`.
