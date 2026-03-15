# Save Session

Saves the current session to a markdown file and updates the CLAUDE.md keyword index.

## Instructions

When the user invokes `/save-session`, do the following:

1. **Identify what was done** in this session. If multiple distinct topics were covered, create separate session files for each.

2. **Determine which project** each topic belongs to:
   - `flexium` - Client library related
   - `flexium-server` - Server related

3. **Generate session file(s)** in `flexium-server/sessions/` with this naming format:
   ```
   sessions/<project>/YYYY-MM-DD-<short-description>.md
   ```
   Example: `sessions/flexium/2026-03-14-colab-setup-gpu-training.md`

4. **Use this structure** for each session file:
   ```markdown
   # Session: YYYY-MM-DD - <Short Description>

   ## Keywords
   <comma-separated keywords for indexing>

   ## Goal
   <One-liner: what we set out to do>

   ## Summary
   <Brief overview of what was accomplished>

   ## What We Did
   - <Key actions/changes>

   ## Issues & Solutions
   | Issue | Solution |
   |-------|----------|
   | ... | ... |

   ## Decisions Made
   - <Any choices we made and why>

   ## Status
   - [x] Completed items
   - [ ] Pending items

   ## Files Changed
   - <List of files modified/created>

   ## Related Sessions
   - <Links to related past sessions, if any>
   ```

5. **Ask about Extra section**: If there's additional context worth saving (resources, environment details, follow-up ideas, blockers, special code snippets), ask the user if they want to include an Extra section.

6. **Confirm each file**: Before updating CLAUDE.md, show the user each session file that will be created and ask if they want to keep it. For each file:
   - Show the filename and a brief summary of what it contains
   - Ask: "Keep this session file? (yes/no/edit)"
   - If "no", skip that file entirely
   - If "edit", ask what changes they want before saving
   - Only proceed with files the user confirms

7. **Update CLAUDE.md**: Update `/home/e155809/views/flexium_project/flexium-server/CLAUDE.md` - add new keywords to the Session Index table, mapping keywords to session file paths. Only add keywords for files the user confirmed in step 6.

8. **Confirm**: Show the user what was saved and the updated keyword index.

9. **Ask about pushing**: Ask the user if they want to commit and push the session files to git. If yes, commit with message format: `docs: Add <session-description> session` and push.
