# Security Policy

## Supported versions

Only the `main` branch is maintained. This is a portfolio / demo project, not a hosted service.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub private vulnerability reporting
("Security" tab, "Report a vulnerability") or email julesbarth13@gmail.com.
Expect a first answer within 7 days.

## Deployment notes

The backend (`backend/server.py`) has no user authentication. Before exposing it beyond localhost:

| Setting | Default | Effect |
|---------|---------|--------|
| `ADMIN_TOKEN` | unset | `POST /rebuild-index` returns 403 until set; then it requires the header `X-Admin-Token: <value>` |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated origins allowed to call the API from a browser. The Next.js UI calls the API server-side (`webapp/src/app/api/ask/route.ts`), so it does not need CORS. |

Other known limits:

- `/ask` and `/ask/stream` have no rate limiting. Put a reverse proxy in front for public use.
- The company documents in `company_data/` are fictional (TechCorp Solutions). Do not index real HR or personal data without access control.
