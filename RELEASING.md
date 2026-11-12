# Releasing

A release is one click. The workflow picks the next version, tests and builds it, waits for your approval, publishes to PyPI, then creates the tag and the GitHub Release.

The git tag is the only source of the version: tag `v0.2.0` is `whatmodel 0.2.0`. Nothing in the repo holds a version number, and no API token or secret is stored anywhere.

## Cut a release

1. Make sure `main` is green in the Actions tab.
2. Open https://github.com/ShekharBhardwaj/whatmodel/actions/workflows/release.yml
3. Click "Run workflow", leave the branch on `main`, and choose what to raise:

   | Choice | Example | Use it for |
   |---|---|---|
   | `patch` | 0.1.0 to 0.1.1 | Fixes and new tokens |
   | `minor` | 0.1.1 to 0.2.0 | New features |
   | `major` | 0.2.0 to 1.0.0 | Breaking changes |

   For the very first release choose `minor`, which gives `0.1.0`.
4. Click the green "Run workflow" button.
5. When the run reaches "Publish to PyPI" it waits for you. Open the run, click "Review deployments", tick `pypi`, and approve.
6. Check the result:

   ```bash
   uvx whatmodel@latest --version
   ```

The same thing from a terminal:

```bash
gh workflow run release.yml -f bump=patch
```

Release notes are generated from merged pull requests, grouped by the labels in `.github/release.yml`. Update `CHANGELOG.md` before you click if you keep it by hand.

### Rehearsals

Tick "Rehearsal only" when you run the workflow. It does everything except publish to PyPI: it tests, builds, and creates a pre-release on GitHub tagged `rcN`, such as `v0.2.0rc1`, with the built files attached. Running it again for the same version gives `rc2`, and so on.

A rehearsal does not exercise the PyPI upload itself, so the first real release is also the first test of trusted publishing.

### Releasing by hand

Publishing a GitHub Release yourself still works and runs the same jobs. Create it with a new tag such as `v0.2.0`; the workflow then publishes that version and attaches the files.

### Tag rules

The tag must be `vX.Y.Z` or `vX.Y.ZrcN`. Anything else, such as `0.2.0`, `v0.2` or `v0.2.0-rc1`, fails in the first job before anything is built.

## What the workflow does

`.github/workflows/release.yml` runs these jobs in order. Each one needs the one before it.

| Job | What it does |
|---|---|
| `plan` | Works out the next tag from the latest one and your choice. Fails fast if the run is not on `main` or the tag is not `vX.Y.Z` or `vX.Y.ZrcN`. |
| `test` | Runs the same checks as CI, on Python 3.10 to 3.13 across Linux, macOS and Windows. |
| `build` | Runs `uv build`, checks the built version equals the tag, runs `twine check`, installs the wheel in a clean environment and runs `whatmodel --version` and `whatmodel qwen3.5:4b --offline`, then uploads `dist/` as an artifact. |
| `publish-pypi` | Publishes that artifact to PyPI, after your approval. Skipped for rehearsals. |
| `github-release` | Creates the tag and the GitHub Release with generated notes, and attaches the sdist and wheel. |

The publish job never rebuilds. It uploads the exact files the `build` job produced.

The tag is created last, only after PyPI has accepted the files. If anything fails before that, no tag exists and the same version can simply be run again.

## One-time setup

Do this once, before the first release. It needs to be done by hand.

### 1. PyPI: add a pending trusted publisher

The project does not exist on PyPI yet, so the publisher is "pending": it creates the project the first time it is used.

1. Sign in at https://pypi.org and open https://pypi.org/manage/account/publishing/
2. Under "Add a new pending publisher", choose the GitHub tab and fill in:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `whatmodel` |
   | Owner | `ShekharBhardwaj` |
   | Repository name | `whatmodel` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

3. Click "Add".

A pending publisher does not reserve the name. Until the first release goes out, someone else could still register `whatmodel`, so do not leave a long gap between this step and the first release.

### 2. GitHub: create the environment

In the repository, open Settings, then Environments. Create an environment named `pypi`, turn on "Required reviewers", add yourself, and save. This is the approval step that stops a release before it reaches PyPI.

Optional but sensible for `pypi`: under "Deployment branches and tags", choose "Selected branches and tags" and add a tag rule for `v*`, so only release tags can deploy.

The environment name must match the workflow and the publisher exactly: `pypi`.

### 3. GitHub: labels for release notes

Create the labels the release notes use if they do not exist: `tokens`, `feature`, `fix`, `skip-changelog`. `bug` and `enhancement` exist by default.

## If a release fails

- **`plan`, `test` or `build` fails.** Nothing was published and no tag was created. Fix `main` and run the workflow again.
- **`publish-pypi` fails with an "invalid-publisher" error.** The pending publisher on PyPI does not match. Check the owner, repository, workflow file name and environment name, then re-run the failed job.
- **The approval times out.** Re-run the failed jobs from the run page. Nothing was published, so nothing needs cleaning up.
- **`github-release` fails after PyPI succeeded.** The version is on PyPI but has no tag yet. Re-run that job from the run page rather than starting a new run, so the tag lands on the commit that was published.
- **A bad version reached PyPI.** A version number can never be reused. Yank the release on PyPI and ship a new patch version.

## The browser page

`docs/index.html` loads the released package from PyPI, so it picks up each new release by itself. To publish the page, turn on GitHub Pages once: repository Settings, Pages, Source "Deploy from a branch", branch `main`, folder `/docs`. It is then served at <https://shekharbhardwaj.github.io/whatmodel/>.

To try the page against a local build, serve a folder that holds `index.html` and a wheel from `uv build`, and open `index.html?wheel=<URL of the wheel>`.
