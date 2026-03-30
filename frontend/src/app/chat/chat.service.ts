import { Injectable, computed, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, from, switchMap, throwError } from 'rxjs';

export type Role = 'user' | 'assistant';

export interface Message {
  role: Role;
  content: string;
  timestamp: Date;
}

export interface ChatRequest {
  prompt: string;
  model: string;
}

export interface ChatResponse {
  response: string;
  model: string;
}

export interface HealthResponse {
  status: 'ok' | 'ollama_unreachable';
  models: string[];
}

export interface RagIngestResponse {
  status: string;
  source: string;
  chunks?: number;
}

export interface RagQueryResponse {
  response: string;
  model: string;
  contexts: Array<{ source: string; score: number }>;
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly apiUrl = 'http://localhost:8000';
  private readonly wsUrl = this.apiUrl.replace(/^http/, 'ws');

  readonly messages = signal<Message[]>([]);
  readonly isLoading = signal(false);
  readonly isStreaming = signal(false);
  readonly models = signal<string[]>(['llama3']);
  readonly ollamaOk = signal(true);

  readonly lastReply = computed(() => {
    const msgs = this.messages();
    return msgs.findLast((m: Message) => m.role === 'assistant')?.content ?? '';
  });

  constructor(private http: HttpClient) {
    this.checkHealth();
  }

  checkHealth(): void {
    this.http.get<HealthResponse>(`${this.apiUrl}/health`).subscribe({
      next: (res) => {
        this.ollamaOk.set(res.status === 'ok');
        if (res.models.length) this.models.set(res.models);
      },
      error: () => this.ollamaOk.set(false),
    });
  }

  private toBase64(buffer: ArrayBuffer): string {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;

    for (let index = 0; index < bytes.length; index += chunkSize) {
      const chunk = bytes.subarray(index, index + chunkSize);
      binary += String.fromCharCode(...chunk);
    }

    return btoa(binary);
  }

  sendMessage(prompt: string, model: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, { prompt, model });
  }

  queryKnowledge(question: string, model: string): Observable<RagQueryResponse> {
    return this.http.post<RagQueryResponse>(`${this.apiUrl}/rag/query`, {
      question,
      model,
      embedding_model: 'all-minilm',
      top_k: 4,
      min_score: 0.2,
    });
  }

  streamMessage(
    prompt: string,
    model: string,
    onToken: (token: string) => void,
    onDone: () => void,
    onError: (err: string) => void,
  ): void {
    const socket = new WebSocket(`${this.wsUrl}/chat/ws`);
    let finished = false;

    socket.onopen = () => {
      socket.send(JSON.stringify({ prompt, model }));
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as { token?: string; error?: string; done?: boolean };

        if (data.error) {
          finished = true;
          onError(data.error);
          socket.close();
          return;
        }

        if (data.token) {
          onToken(data.token);
        }

        if (data.done) {
          finished = true;
          onDone();
          socket.close();
        }
      } catch (err) {
        finished = true;
        onError(String(err));
        socket.close();
      }
    };

    socket.onerror = () => {
      if (!finished) {
        finished = true;
        onError('WebSocket connection failed');
      }
    };

    socket.onclose = () => {
      if (!finished) {
        finished = true;
        onDone();
      }
    };
  }

  multiTurnChat(model: string): Observable<{ response: string; model: string }> {
    const messages = this.messages().map((m) => ({
      role: m.role,
      content: m.content,
    }));

    return this.http.post<{ response: string; model: string }>(
      `${this.apiUrl}/chat/multi`,
      { messages, model },
    );
  }

  uploadDocument(file: File): Observable<RagIngestResponse> {
    const extension = file.name.includes('.') ? file.name.split('.').pop()?.toLowerCase() ?? '' : '';
    const allowedExtensions = new Set(['txt', 'md', 'json', 'csv', 'log', 'pdf']);
    if (!allowedExtensions.has(extension)) {
      return throwError(() => new Error(
        'Supported files: .txt, .md, .json, .csv, .log, .pdf',
      ));
    }

    const payload = {
      source: file.name,
      metadata: {
        size: file.size,
        type: file.type || 'text/plain',
      },
    };

    if (extension === 'pdf') {
      return from(file.arrayBuffer()).pipe(
        switchMap((buffer) => this.http.post<RagIngestResponse>(`${this.apiUrl}/rag/ingest`, {
          ...payload,
          content: '',
          content_base64: this.toBase64(buffer),
        })),
      );
    }

    return from(file.text()).pipe(
      switchMap((content) => this.http.post<RagIngestResponse>(`${this.apiUrl}/rag/ingest`, {
        ...payload,
        content,
      })),
    );
  }

  addMessage(role: Role, content: string): void {
    this.messages.update((msgs) => [
      ...msgs,
      { role, content, timestamp: new Date() },
    ]);
  }

  updateLastAssistantMessage(append: string): void {
    this.messages.update((msgs) => {
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
