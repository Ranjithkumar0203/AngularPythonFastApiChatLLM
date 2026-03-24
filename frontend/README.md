# Angular 20 · Ollama Chat

A full Angular 20 standalone app that integrates with a FastAPI + Ollama backend.

## Features

- ✅ **Angular 20** — standalone components, Signals, `@for` / `@if` control flow
- ✅ **Signals** — all state managed with `signal()`, `computed()`, `update()`  
- ✅ **OnPush** change detection on every component
- ✅ **Lazy-loaded** chat route
- ✅ **3 chat modes**: single-shot, streaming (SSE), multi-turn
- ✅ `provideHttpClient(withFetch())` — uses Fetch API instead of XHR

## Project Structure

```
src/
├── main.ts                     # Bootstrap (standalone)
├── index.html
├── styles.scss                 # Global styles
└── app/
    ├── app.component.ts        # Root shell with <router-outlet>
    ├── app.config.ts           # provideRouter, provideHttpClient, provideAnimations
    ├── app.routes.ts           # Lazy-loads ChatComponent
    └── chat/
        ├── chat.service.ts     # Signal-based state + API calls
        ├── chat.component.ts   # OnPush standalone component
        ├── chat.component.html # @for/@if control flow
        ├── chat.component.scss # Scoped styles
        └── chat.component.spec.ts
```

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Start Ollama
ollama serve
ollama pull llama3

# 3. Start FastAPI backend (from /backend folder)
pip install fastapi uvicorn ollama
uvicorn main:app --reload --port 8000

# 4. Start Angular dev server
ng serve
# → http://localhost:4200
```

## Angular 20 Highlights Used

| Feature | Where |
|---|---|
| `signal()` / `computed()` | `chat.service.ts`, `chat.component.ts` |
| `signal.update()` | `updateLastAssistantMessage()` in service |
| `@for` / `@if` control flow | `chat.component.html` |
| `ChangeDetectionStrategy.OnPush` | `chat.component.ts` |
| `provideHttpClient(withFetch())` | `app.config.ts` |
| Lazy-loaded standalone component | `app.routes.ts` |
| `linkedSignal` / `effect` | Ready to add in service |
