import { Injectable, signal, computed } from '@angular/core';
import { HttpClient }                   from '@angular/common/http';
import { Observable }                   from 'rxjs';

// ── Types ──────────────────────────────────────────────────────────────────────
export type Role = 'user' | 'assistant';

export interface Message {
  role: Role;
  content: string;
  timestamp: Date;
}

export interface ChatRequest  { prompt: string; model: string; }
export interface ChatResponse { response: string; model: string; }

export interface HealthResponse {
  status: 'ok' | 'ollama_unreachable';
  models: string[];
}

// ── Service ────────────────────────────────────────────────────────────────────
@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly apiUrl = 'http://localhost:8000';

  // ── Angular 20 Signals for reactive state ──────────────────────────────────
  readonly messages   = signal<Message[]>([]);
  readonly isLoading  = signal(false);
  readonly isStreaming = signal(false);
  readonly models     = signal<string[]>(['llama3']);
  readonly ollamaOk   = signal(true);

  // Computed: last message from assistant
  readonly lastReply = computed(() => {
    const msgs = this.messages();
    return msgs.findLast((m: Message) => m.role === 'assistant')?.content ?? '';
  });

  constructor(private http: HttpClient) {
    this.checkHealth();
  }

  // ── Health check ────────────────────────────────────────────────────────────
  checkHealth(): void {
    this.http.get<HealthResponse>(`${this.apiUrl}/health`).subscribe({
      next: (res) => {
        this.ollamaOk.set(res.status === 'ok');
        if (res.models.length) this.models.set(res.models);
      },
      error: () => this.ollamaOk.set(false),
    });
  }

  // ── Non-streaming: full response via HttpClient ─────────────────────────────
  sendMessage(prompt: string, model: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, { prompt, model });
  }

  // ── Streaming: SSE via native fetch + ReadableStream ────────────────────────
  streamMessage(
    prompt: string,
    model: string,
    onToken: (token: string) => void,
    onDone: () => void,
    onError: (err: string) => void,
  ): void {
    fetch(`${this.apiUrl}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, model }),
    })
      .then(async (res) => {
        if (!res.ok || !res.body) {
          onError(`HTTP ${res.status}`);
          return;
        }
        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            // Flush any remaining complete event when stream closes.
            if (buffer.trim().length > 0) {
              const frame = buffer.trim();
              if (frame.startsWith('data: ')) {
                const data = JSON.parse(frame.slice(6)) as { token?: string; error?: string };
                if (data.error) { onError(data.error); return; }
                if (data.token) onToken(data.token);
              }
            }
            onDone();
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split('\n\n');
          buffer = frames.pop() ?? '';

          for (const frame of frames) {
            const line = frame.trim();
            if (!line.startsWith('data: ')) continue;
            const data = JSON.parse(line.slice(6)) as { token?: string; error?: string };
            if (data.error) { onError(data.error); return; }
            if (data.token) onToken(data.token);
          }
        }
      })
      .catch(err => onError(String(err)));
  }

  // ── Multi-turn chat ─────────────────────────────────────────────────────────
  multiTurnChat(model: string): Observable<{ response: string; model: string }> {
    const messages = this.messages().map(m => ({
      role: m.role,
      content: m.content,
    }));
    return this.http.post<{ response: string; model: string }>(
      `${this.apiUrl}/chat/multi`,
      { messages, model },
    );
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────
  addMessage(role: Role, content: string): void {
    this.messages.update(msgs => [
      ...msgs,
      { role, content, timestamp: new Date() },
    ]);
  }

  updateLastAssistantMessage(append: string): void {
    this.messages.update(msgs => {
      const copy = [...msgs];
      const last = copy.findLastIndex((m: Message) => m.role === 'assistant');
      if (last !== -1) copy[last] = { ...copy[last], content: copy[last].content + append };
      return copy;
    });
  }

  clearMessages(): void {
    this.messages.set([]);
  }
}
