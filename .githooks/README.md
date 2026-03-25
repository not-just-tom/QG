This directory contains repository-tracked Git hooks.

To enable these hooks for your local clone, run:

  git config core.hooksPath .githooks

The included `pre-commit` hook will block commits that stage any file larger than 100 MB and will append those file paths to the repository `.gitignore` so they are not accidentally committed again.

If you want to track large files, consider installing Git LFS: https://git-lfs.github.com/
