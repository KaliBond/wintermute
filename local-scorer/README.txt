CAMS Scorer — local edition
============================

Score any country, company, or city on the eight CAMS nodes, using
your own OpenAI account. Nothing is sent to Neural Nations — this
runs entirely on your machine and talks directly to OpenAI.

Three versions are provided:
   cams-scorer-gui.exe   Windows app, double-click to open, no command line
   cams-scorer.exe       Windows command-line version
   cams-scorer-linux     Linux / WSL command-line version

GUI VERSION (cams-scorer-gui.exe) — easiest
--------------------------------------------
1. Double-click cams-scorer-gui.exe to open it.
2. On first launch it asks for your OpenAI API key
   (get one at https://platform.openai.com/api-keys). It's saved
   locally in %APPDATA%\CAMSScorer\config.json (plain text) so you
   won't be asked again.
3. Fill in the form (what to score, name, years, number of passes)
   and click "Run scoring pass".
4. Results appear in the table and are saved as CSV files in
   Documents\CAMS Scorer Output — click "Open output folder" to find
   them.

COMMAND-LINE VERSIONS — setup (one-time)
------------------------------------------
1. Get an OpenAI API key: https://platform.openai.com/api-keys
2. Set it as an environment variable before running:

   Windows (PowerShell):
       $env:OPENAI_API_KEY = "sk-..."

   Windows (cmd.exe):
       set OPENAI_API_KEY=sk-...

   Linux / WSL:
       export OPENAI_API_KEY="sk-..."
       chmod +x cams-scorer-linux   (first time only)

USAGE (command-line versions)
------------------------------
   cams-scorer.exe <country|company|city> <entity> <start_year> [end_year] [options]     (Windows)
   ./cams-scorer-linux <country|company|city> <entity> <start_year> [end_year] [options]  (Linux/WSL)

Examples:
   cams-scorer.exe country USA 2020 2024
   cams-scorer.exe company Tesla 2023
   cams-scorer.exe city Detroit 2019 2023 --passes 5

Options:
   --passes N        number of isolated scoring passes (default 5)
   --min-passes N    minimum valid passes required per cell (default: min(3, passes))
   --model NAME      OpenAI model to use (default: gpt-5, or set OPENAI_MODEL)
   --out PREFIX      output filename prefix (default: derived from entity name)

OUTPUT
------
Two CSV files are written to the current folder:
   <entity>_cams_scores.csv     — ensemble mean per node (Coherence, Capacity, Stress, Abstraction)
   <entity>_cams_envelope.csv   — standard deviation + pass count per cell (uncertainty)

NOTES
-----
- Each pass is a separate live model call — a 5-pass run typically takes several minutes.
- This performs raw scoring only (no Node Value, Bond Strength, or diagnosis).
  See neuralnations.org for the full CAMS calculation pipeline and published datasets.
- Cost is billed to your own OpenAI account, at your account's usual rates.
