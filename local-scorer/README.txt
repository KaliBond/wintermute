CAMS Scorer — local edition
============================

Score any country, company, or city on the eight CAMS nodes, using your
own OpenAI or Kimi (Moonshot AI) account. Nothing is sent to Neural
Nations — this runs entirely on your machine and talks directly to
whichever provider you choose.

Five versions are provided:
   cams-scorer-gui.exe        Windows app (OpenAI), double-click, no command line
   cams-scorer.exe            Windows command-line version (OpenAI)
   cams-scorer-linux          Linux / WSL command-line version (OpenAI)
   cams-scorer-gui-kimi.exe   Windows app (Kimi), double-click, no command line
   cams-scorer-kimi.exe       Windows command-line version (Kimi)

The Kimi edition is Windows-only for now, and runs are capped at 10
years and 3 passes per request (3 passes carries roughly 85% of the
ensemble signal).

GUI VERSIONS — easiest
------------------------
1. Double-click cams-scorer-gui.exe (OpenAI) or cams-scorer-gui-kimi.exe
   (Kimi) to open it.
2. On first launch it asks for your API key:
     OpenAI — https://platform.openai.com/api-keys
     Kimi   — https://platform.moonshot.ai/console/api-keys
   It's saved locally in plain text — %APPDATA%\CAMSScorer\config.json
   (OpenAI) or %APPDATA%\CAMSScorerKimi\config.json (Kimi) — so you
   won't be asked again.
3. Fill in the form (what to score, name, years, number of passes)
   and click "Run scoring pass".
4. Results appear in the table and are saved as CSV files in
   Documents\CAMS Scorer Output — click "Open output folder" to find
   them.

COMMAND-LINE VERSIONS — setup (one-time)
------------------------------------------
1. Get an API key:
     OpenAI — https://platform.openai.com/api-keys
     Kimi   — https://platform.moonshot.ai/console/api-keys
2. Set it as an environment variable before running:

   Windows (PowerShell):
       $env:OPENAI_API_KEY = "sk-..."     (OpenAI edition)
       $env:KIMI_API_KEY = "sk-..."       (Kimi edition)

   Windows (cmd.exe):
       set OPENAI_API_KEY=sk-...
       set KIMI_API_KEY=sk-...

   Linux / WSL (OpenAI edition only):
       export OPENAI_API_KEY="sk-..."
       chmod +x cams-scorer-linux   (first time only)

USAGE (command-line versions)
------------------------------
   cams-scorer.exe <country|company|city> <entity> <start_year> [end_year] [options]       (Windows, OpenAI)
   cams-scorer-kimi.exe <country|company|city> <entity> <start_year> [end_year] [options]   (Windows, Kimi)
   ./cams-scorer-linux <country|company|city> <entity> <start_year> [end_year] [options]    (Linux/WSL, OpenAI)

Examples:
   cams-scorer.exe country USA 2020 2024
   cams-scorer.exe company Tesla 2023
   cams-scorer.exe city Detroit 2019 2023 --passes 5

   cams-scorer-kimi.exe country USA 2020 2024
   cams-scorer-kimi.exe company Tesla 2023
   cams-scorer-kimi.exe city Detroit 2019 2023 --passes 3

Options:
   --passes N        number of isolated scoring passes (OpenAI default 5; Kimi default and max 3)
   --min-passes N    minimum valid passes required per cell (default: min(3, passes))
   --model NAME      model to use (OpenAI default: gpt-5, or set OPENAI_MODEL;
                      Kimi default: kimi-k3, or set KIMI_MODEL)
   --out PREFIX      output filename prefix (default: derived from entity name)

The Kimi edition also enforces a 10-year cap on the start_year..end_year
range (the OpenAI edition has no server-side cap on the local build).

OUTPUT
------
Two CSV files are written to the current folder:
   <entity>_cams_scores.csv     — ensemble mean per node (Coherence, Capacity, Stress, Abstraction)
   <entity>_cams_envelope.csv   — standard deviation + pass count per cell (uncertainty)

NOTES
-----
- Each pass is a separate live model call — a multi-pass run typically takes several minutes.
- This performs raw scoring only (no Node Value, Bond Strength, or diagnosis).
  See neuralnations.org for the full CAMS calculation pipeline and published datasets.
- Cost is billed to your own account, at your provider's usual rates.
