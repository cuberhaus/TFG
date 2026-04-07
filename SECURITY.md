# Security Policy — TFG

## Reporting a Vulnerability
If you discover a security vulnerability, please email polcg10@gmail.com. Do not open a public issue.

## Security Considerations

### No Authentication (Development Only)
- The FastAPI backend (port 8082) and Vite frontend (port 5173) have **no authentication**.
- This is acceptable for local development only. **Do not deploy to a public network without adding auth.**
- If exposed, anyone can upload files, trigger model inference, and access results.

### File Upload Validation
- The dataset drag-and-drop feature accepts file uploads. This is a high-risk surface:
  - Validate file types — only accept expected formats (images, specific archive types).
  - Check file size limits to prevent resource exhaustion.
  - Do not trust client-provided filenames — sanitize or generate server-side names to prevent path traversal.
  - Store uploads outside the web root and serve them through controlled endpoints.
  - Scan uploaded files for malformed content that could exploit image processing libraries.

### CORS Configuration
- The Vite dev server proxies to FastAPI. In production, configure CORS explicitly.
- Do not use `allow_origins=["*"]` with `allow_credentials=True`.
- Restrict origins to the actual frontend domain.

### Medical Data Handling
- This project processes polyp detection images — potentially **medical/clinical data**.
- Do not use real patient data in development without proper de-identification.
- If handling real medical images, comply with applicable regulations (GDPR, HIPAA-like local equivalents).
- Do not log or expose image data in error messages or API responses.
- Implement data retention and deletion policies.

### Model & Inference Security
- Deep learning models can be adversarially attacked. Be aware that crafted inputs may produce incorrect results.
- Do not serve model weights publicly unless intended — they may encode training data patterns.

### Recommendations
- Add API key or session-based authentication before any shared deployment.
- Use HTTPS for all communication, especially when transmitting medical images.
- Pin Python dependencies and run `pip audit` to check for known vulnerabilities.
- Bind services to `127.0.0.1` for local-only access.
