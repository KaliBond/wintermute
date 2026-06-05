/* ================================================================
   ENS CSV loader for CAMS Interpreter
   Fetches all data/nations/*_ENS.csv files and returns them in
   the same shape as cams-data.js: { [nation][year][node]: [C,K,S,A] }
   ================================================================ */

const ENS_PATHS = {
  Argentina:  '../data/nations/Argentina_ENS.csv',
  Australia:  '../data/nations/Australia_ENS.csv',
  Canada:     '../data/nations/Canada_ENS.csv',
  Chile:      '../data/nations/Chile_ENS.csv',
  China:      '../data/nations/China_ENS.csv',
  Colombia:   '../data/nations/Colombia_ENS.csv',
  France:     '../data/nations/France_ENS.csv',
  Germany:    '../data/nations/Germany_ENS.csv',
  India:      '../data/nations/India_ENS.csv',
  Iran:       '../data/nations/Iran_ENS.csv',
  Japan:      '../data/nations/Japan_ENS.csv',
  Norway:     '../data/nations/Norway_ENS.csv',
  Poland:     '../data/nations/Poland_ENS.csv',
  Russia:     '../data/nations/Russia_ENS.csv',
  Sweden:     '../data/nations/Sweden_ENS.csv',
  Thailand:   '../data/nations/Thailand_ENS.csv',
  Turkiye:    '../data/nations/Turkiye_ENS.csv',
  UK:         '../data/nations/UK_ENS.csv',
  USA:        '../data/nations/USA_ENS.csv',
};

function parseENS(text) {
  const lines = text.trim().split('\n');
  const header = lines[0].split(',').map(h => h.trim().toLowerCase());
  const yIdx = header.indexOf('year');
  const nIdx = header.indexOf('node');
  const cIdx = header.indexOf('coherence');
  const kIdx = header.indexOf('capacity');
  const sIdx = header.indexOf('stress');
  const aIdx = header.indexOf('abstraction');
  const result = {};
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',');
    if (cols.length < 6) continue;
    const year = parseInt(cols[yIdx]);
    const node = cols[nIdx] && cols[nIdx].trim();
    if (!year || !node) continue;
    if (!result[year]) result[year] = {};
    result[year][node] = [
      parseFloat(cols[cIdx]),
      parseFloat(cols[kIdx]),
      parseFloat(cols[sIdx]),
      parseFloat(cols[aIdx]),
    ];
  }
  return result;
}

async function loadAllENS() {
  const entries = await Promise.all(
    Object.entries(ENS_PATHS).map(async ([name, path]) => {
      try {
        const resp = await fetch(path);
        if (!resp.ok) return null;
        const text = await resp.text();
        return [name, parseENS(text)];
      } catch (e) {
        console.warn('ENS load failed:', name, e);
        return null;
      }
    })
  );
  const result = {};
  for (const entry of entries) {
    if (entry) result[entry[0]] = entry[1];
  }
  return result;
}
