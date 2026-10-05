# Security

The current 0.2 series is the maintained code path. Legacy scripts and pickle artifacts are unsupported.

Do not include credentials, patient records, private PDFs, extracted passages, or database dumps in public issues. Report vulnerabilities through the repository's private vulnerability reporting channel when available; if it is unavailable, contact the maintainer privately before disclosing sensitive details.

The API requires a shared key by default and returns passages only through authenticated POST requests. Health checks expose no source passages. Shared-key authentication does not provide tenant isolation or per-document permissions. Use HTTPS, gateway rate limits, and deployment access controls for remote serving.

PDF ingestion is an offline administrative operation. Process untrusted PDFs in an isolated environment; parser dependencies and resource limits are not a security sandbox. Keep source material and the SQLite corpus protected on disk.

No runtime corpus path loads pickle objects. Numeric vector caches are validated, model revisions are pinned, safetensors are required for model weights, and remote model code is disabled. These measures do not replace dependency updates or review of externally supplied models and documents.

The service is an engineering retrieval foundation and does not claim healthcare compliance, clinical validation, or suitability for diagnosis or treatment.
