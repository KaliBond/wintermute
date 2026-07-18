/* ──────────────────────────────────────────────────────────────────────────
   CAMS Scorer — live raw scoring against the Neural Nations CAMS API.
   The API key never reaches this page: the Fly.io backend (cams-raw-scorer)
   holds KIMI_API_KEY server-side and does the Kimi (Moonshot) calls itself.
   This page only sends the site token (below), which is a soft gate, not a
   secret — it ships in public source, so the backend also rate-limits by
   IP. Only the "country" variant is served live; company/city are gated to
   a sales inquiry. See CAMS_RAW_SCORER_SUITE/NEURALNATIONS_API.md for the
   full contract.
   ────────────────────────────────────────────────────────────────────────── */
(function () {
  "use strict";

  var API_ENDPOINT = "https://cams-raw-scorer.fly.dev/api/cams/raw-run";
  var API_TOKEN = "EVW3r-pPd2Kugpl1TsvDaijJ1HzdepNZDwMiNm3WdUg";

  var form = document.getElementById("cs-form");
  var runBtn = document.getElementById("cs-run");
  var statusEl = document.getElementById("cs-status");
  var resultsCard = document.getElementById("cs-results");
  var resultsTitle = document.getElementById("cs-results-title");
  var skel = document.getElementById("cs-skel");
  var table = document.getElementById("cs-table");
  var dlBtn = document.getElementById("cs-dl-mean");

  var lastMeanCsv = "";

  function setStatus(text, isErr) {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("cs-err", !!isErr);
  }

  function parseCsv(text) {
    var lines = text.trim().split(/\r?\n/);
    if (!lines.length) return { header: [], rows: [] };
    var header = lines[0].split(",");
    var rows = lines.slice(1).filter(Boolean).map(function (l) { return l.split(","); });
    return { header: header, rows: rows };
  }

  function renderTable(csvText) {
    var parsed = parseCsv(csvText);
    var thead = "<thead><tr>" + parsed.header.map(function (h) { return "<th>" + h + "</th>"; }).join("") + "</tr></thead>";
    var tbody = "<tbody>" + parsed.rows.map(function (r) {
      return "<tr>" + r.map(function (c, i) {
        var numeric = i >= 3; // Entity, Year, Node are text; the rest are numeric
        return "<td class=\"" + (numeric ? "cs-num" : "") + "\">" + c + "</td>";
      }).join("") + "</tr>";
    }).join("") + "</tbody>";
    table.innerHTML = thead + tbody;
  }

  function csvToDownloadUrl(csvText) {
    var blob = new Blob([csvText], { type: "text/csv" });
    return URL.createObjectURL(blob);
  }

  function run(e) {
    e.preventDefault();
    var variant = document.getElementById("cs-variant").value;
    var entity = document.getElementById("cs-entity").value.trim();
    var passes = +document.getElementById("cs-passes").value;
    var startYear = +document.getElementById("cs-start").value;
    var endRaw = document.getElementById("cs-end").value.trim();
    var endYear = endRaw ? +endRaw : startYear;

    if (variant !== "country") { setStatus("Company and city scoring are by inquiry — email sales@neuralnations.org.", true); return; }
    if (!entity) { setStatus("Enter an entity to score.", true); return; }
    if (!startYear) { setStatus("Enter a start year.", true); return; }
    if (endYear - startYear + 1 > 10) { setStatus("Runs are capped at 10 years — narrow the range.", true); return; }

    runBtn.disabled = true;
    setStatus("Running " + passes + " isolated passes, one at a time — this can take a few minutes…");
    resultsCard.style.display = "";
    skel.classList.add("on");
    table.innerHTML = "";

    fetch(API_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        variant: variant,
        entity: entity,
        start_year: startYear,
        end_year: endYear,
        n_passes: passes,
        include_passes: false,
        api_token: API_TOKEN
      })
    }).then(function (r) {
      if (!r.ok) return r.text().then(function (t) { throw new Error(t || ("HTTP " + r.status)); });
      return r.json();
    }).then(function (data) {
      lastMeanCsv = data.mean_csv || "";
      resultsTitle.textContent = data.entity + " · " + data.start_year + (data.end_year !== data.start_year ? "–" + data.end_year : "") + " (" + data.model + ")";
      renderTable(lastMeanCsv);
      dlBtn.href = csvToDownloadUrl(lastMeanCsv);
      dlBtn.download = data.entity.replace(/\s+/g, "_") + "_cams_scores.csv";
      setStatus("Done — " + passes + " passes ensembled.");
    }).catch(function (err) {
      var msg = String(err && err.message || err);
      if (msg.indexOf("429") !== -1 || /rate limit/i.test(msg)) {
        setStatus("Rate limit reached — try again in a bit.", true);
      } else {
        setStatus("Scoring failed: " + msg, true);
      }
      resultsCard.style.display = "none";
    }).then(function () {
      runBtn.disabled = false;
      skel.classList.remove("on");
    });
  }

  form.addEventListener("submit", run);
})();
