# Changelog

## 0.2.0

- Rename the project and repository to Medical RAG Engine / `medical-rag-engine`.
- Replace the prototype script layout with a modular, installable Python package.
- Remove tracked environments, bytecode, generated chapter outputs, and pickle caches.
- Standardize chapter boundaries and citations on one-based physical PDF pages.
- Add transactional SQLite corpus updates and validated numeric embedding persistence.
- Fix stale cache reuse, textbook-filter recall, and invalid FAISS result handling.
- Add a scriptable JSON CLI and an authenticated read-only search API with bounded bodies.
- Pin dependencies and the default model; use CPU-only PyTorch and safetensors.
- Add regression tests, real-model verification, cross-platform CI, and container smoke testing.
- Rewrite onboarding, architecture, operations, security, and migration documentation.

This release changes the storage format and command interface. Reingest source PDFs as described in [migration guidance](docs/migration.md).
