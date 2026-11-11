# Releasing

Publishing a GitHub Release publishes that version to PyPI. The git tag is the only source of the version: tag `v0.2.0` ships `whatmodel 0.2.0`. Nothing in the repo holds a version number, and no API token or secret is stored anywhere.

## Cut a release

1. Make sure `main` is green in the Actions tab.
2. Update `CHANGELOG.md`: rename `Unreleased` to the new version and date, add a fresh `Unreleased` section, and fix the links at the bottom. Commit and push to `main`.
3. Open https://github.com/ShekharBhardwaj/whatmodel/releases/new
4. In "Choose a tag", type the new tag, for example `v0.2.0`, and pick "Create new tag on publish". Target `main`.
5. Click "Generate release notes". The categories come from `.github/release.yml`.
6. Click "Publish release".
7. Watch the "Release" workflow in the Actions tab. When it reaches `publish-pypi` it waits for you: open the run, click "Review deployments", tick `pypi`, and approve.
8. Check the result:

   ```bash
   uvx whatmodel@0.2.0 --version
   ```

The same thing from the command line, after step 2:

```bash
gh release create v0.2.0 --target main --generate-notes
```

### Release candidates

A tag ending in `rcN`, such as `v0.2.0rc1`, is a rehearsal. It is tested, built and attached to the GitHub Release, but the PyPI job is skipped, so nothing is published. Tick "Set as a pre-release" when you publish it.

Use one to check the pipeline before a real release. It does not exercise the PyPI upload itself, so the first real release is also the first test of trusted publishing. To try the built wheel, download it from the release page and run:

```bash
uvx --from ./whatmodel-0.2.0rc1-py3-none-any.whl whatmodel --version
```

### Tag rules

The tag must be `vX.Y.Z` or `vX.Y.ZrcN`. Anything else, such as `0.2.0`, `v0.2` or `v0.2.0-rc1`, fails in the first job before anything is built.

## What the workflow does

`.github/workflows/release.yml` runs these jobs in order. Each one needs the one before it.

| Job | What it does |
|---|---|
| `check-tag` | Fails fast if the tag is not `vX.Y.Z` or `vX.Y.ZrcN`. |
| `test` | Runs the same checks as CI, on Python 3.10 to 3.13 across Linux, macOS and Windows. |
| `build` | Runs `uv build`, checks the built version equals the tag, runs `twine check`, installs the wheel in a clean environment and runs `whatmodel --version` and `whatmodel qwen3.5:4b --offline`, then uploads `dist/` as an artifact. |
| `publish-pypi` | Publishes that artifact to PyPI, after your approval. Skipped for `rc` tags. |
| `attach-assets` | Uploads the sdist and wheel to the GitHub Release page. |

The publish job never rebuilds. It uploads the exact files the `build` job produced.

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

- **`check-tag` fails.** Delete the release and the tag, then publish again with a valid tag.
- **`test` or `build` fails.** Nothing was published. Fix `main`, delete the release and tag, and publish the release again.
- **`publish-pypi` fails with an "invalid-publisher" error.** The pending publisher on PyPI does not match. Check the owner, repository, workflow file name and environment name, then re-run the failed job.
- **The approval times out.** Re-run the failed jobs from the run page. Nothing was published, so nothing needs cleaning up.
- **A bad version reached PyPI.** A version number can never be reused. Yank the release on PyPI and ship a new patch version.
