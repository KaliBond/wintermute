CAMS Scorer — local edition (universal)
=========================================

Score any country, company, or city on the eight CAMS nodes, using your
own account with the provider of your choice: OpenAI, Grok (xAI), Claude
(Anthropic), or Kimi (Moonshot AI). Nothing is sent to Neural Nations —
this runs entirely on your machine and talks directly to whichever
provider you pick.

Three versions are provided:
   cams-scorer-gui-universal.exe   Windows app, double-click, pick a provider from a dropdown
   cams-scorer-universal.exe       Windows command-line version, provider is the first argument
   cams-scorer-linux               Linux / WSL command-line version — older single-provider (OpenAI) build

The universal edition (Windows) supports all four providers and is
capped at a 9-year range and a choice of 1, 3, or 5 passes per request
(3 passes carries roughly 85% of the ensemble signal, so it's a
reasonable middle ground; 5 is the most reliable, 1 the fastest). The
Linux binary is the older OpenAI-only build and does not share these
caps or the provider choice.

GUI VERSION — easiest
------------------------
1. Double-click cams-scorer-gui-universal.exe to open it.
2. Pick a provider from the dropdown: OpenAI, Grok, Claude, or Kimi.
3. The first time you use a given provider, it asks for that provider's
   API key:
     OpenAI — https://platform.openai.com/api-keys
     Grok   — https://console.x.ai
     Claude — https://console.anthropic.com/settings/keys
     Kimi   — https://platform.moonshot.ai/console/api-keys
   Each provider's key is saved separately, in plain text, at
   %APPDATA%\CAMSScorerUniversal\config.json, so switching providers
   later won't lose the others.
4. Fill in the form (what to score, name, years, number of passes —
   1, 3, or 5) and click "Run scoring pass".
5. Results appear in the table and are saved as CSV files in
   Documents\CAMS Scorer Output — click "Open output folder" to find
   them.

COMMAND-LINE VERSION — setup (one-time)
------------------------------------------
1. Get an API key for whichever provider you want (see URLs above).
2. Set the matching environment variable before running:

   Windows (PowerShell):
       $env:OPENAI_API_KEY = "sk-..."       (for the openai provider)
       $env:XAI_API_KEY = "xai-..."         (for the grok provider)
       $env:ANTHROPIC_API_KEY = "sk-ant-..." (for the claude provider)
       $env:KIMI_API_KEY = "sk-..."         (for the kimi provider)

   Windows (cmd.exe):
       set OPENAI_API_KEY=sk-...
       set XAI_API_KEY=xai-...
       set ANTHROPIC_API_KEY=sk-ant-...
       set KIMI_API_KEY=sk-...

USAGE
-----
   cams-scorer-universal.exe <provider> <country|company|city> <entity> <start_year> [end_year] [options]

   <provider> is one of: openai, grok, claude, kimi

Examples:
   cams-scorer-universal.exe openai country USA 2020 2024
   cams-scorer-universal.exe grok company Tesla 2023
   cams-scorer-universal.exe claude city Detroit 2019 2023 --passes 2
   cams-scorer-universal.exe kimi country Japan 2023

Options:
   --passes N        number of isolated scoring passes, one of 1, 3, 5 (default 3)
   --min-passes N    minimum valid passes required per cell (default: passes)
   --model NAME      model override (default per provider: gpt-5 for OpenAI,
                      grok-4.5 for Grok, claude-sonnet-5 for Claude, kimi-k3 for Kimi)
   --out PREFIX      output filename prefix (default: derived from entity name)

Runs are capped at a 9-year range (start_year..end_year) regardless of
provider, and passes must be 1, 3, or 5.

OUTPUT
------
Two CSV files are written to the current folder:
   <entity>_cams_scores.csv     — ensemble mean per node (Coherence, Capacity, Stress, Abstraction)
   <entity>_cams_envelope.csv   — standard deviation + pass count per cell (uncertainty)

LINUX / WSL (older, OpenAI-only build)
-----------------------------------------
   export OPENAI_API_KEY="sk-..."
   chmod +x cams-scorer-linux   (first time only)
   ./cams-scorer-linux <country|company|city> <entity> <start_year> [end_year] [--passes N]

This build predates the universal edition: no provider choice, no
9-year/1-3-5-pass cap, default 5 passes.

NOTES
-----
- Each pass is a separate live model call — a 3-pass run typically takes a couple of minutes.
- This performs raw scoring only (no Node Value, Bond Strength, or diagnosis).
  See neuralnations.org for the full CAMS calculation pipeline and published datasets.
- Cost is billed to your own account, at your provider's usual rates.
