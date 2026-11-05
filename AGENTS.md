# Project Architecture & Porting Rules

## Repository Roles
- **Target Repo (`./bg-cnf-fm-condensed-matter`)**: The clean, public-facing Boltzmann generator implementation. ALL code edits and new files must happen here.
- **Reference Repo (`../bg_crystals`)**: Read-only reference containing experimental features. NEVER modify files in this directory, and NEVER copy dead-end experiments unless explicitly instructed.

## Code Standards
- Maintain the clean architecture of the Target Repo.
- Strip out hardcoded debugging flags, unused imports, and experimental logging when porting modules from the Reference Repo.
- Ensure all ported normalizing flow / energy modules include type hints and docstrings matching the Target Repo style.