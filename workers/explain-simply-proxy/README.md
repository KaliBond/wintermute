# explain-simply-proxy

Cloudflare Worker that holds the Kimi/Moonshot API key server-side and
forwards chat-completion requests from the site's "Explain Simply"
feature (`explain-simply.js`). Restricts origin to neuralnations.org,
overrides the model server-side, and caps prompt/response size.

## Deploy

```
npm install -g wrangler   # if not already installed
cd workers/explain-simply-proxy
wrangler login
wrangler secret put KIMI_API_KEY
# paste the ROTATED key when prompted — never commit it, never paste it into chat
wrangler deploy
```

`wrangler deploy` prints the live URL, e.g.
`https://explain-simply-proxy.<your-subdomain>.workers.dev`.

## Point the site at it

In `index.html`, `start-here.html`, and `explore.html`, set:

```html
<script>
window.EXPLAIN_SIMPLY_CONFIG = {
  endpoint: 'https://explain-simply-proxy.<your-subdomain>.workers.dev',
  model: 'kimi-k2-0905-preview'   // cosmetic only — the Worker enforces the real model
};
</script>
```

No `apiKey` field — the key never leaves the Worker.

## Rotating the key later

```
wrangler secret put KIMI_API_KEY
```
No site changes needed; the endpoint URL stays the same.
