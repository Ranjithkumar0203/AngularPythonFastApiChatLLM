import {
  Component, inject, signal, computed,
  ViewChild, ElementRef, AfterViewChecked,
  ChangeDetectionStrategy,
} from '@angular/core';
import { CommonModule }  from '@angular/common';
import { FormsModule }   from '@angular/forms';
import { ChatService }   from './chat.service';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss',
  // Angular 20: OnPush is recommended default for new components
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ChatComponent implements AfterViewChecked {
  @ViewChild('messagesEnd') private messagesEnd!: ElementRef<HTMLDivElement>;

  readonly chat = inject(ChatService);

  // ── Local UI signals ───────────────────────────────────────────────────────
  prompt     = signal('');
  model      = signal('qwen2.5:7b');
  mode       = signal<'stream' | 'multi' | 'single'>('stream');
  errorMsg   = signal('');
  uploadName = signal('');
  uploadMsg  = signal('');
  isUploading = signal(false);

  // Computed helpers exposed to template
  canSend = computed(() =>
    this.prompt().trim().length > 0 &&
    !this.chat.isLoading() &&
    !this.chat.isStreaming()
  );

  ngAfterViewChecked(): void {
    this.messagesEnd?.nativeElement.scrollIntoView({ behavior: 'smooth' });
  }

  send(): void {
    const text  = this.prompt().trim();
    const mdl   = this.model();
    if (!text) return;

    this.errorMsg.set('');
    this.chat.addMessage('user', text);
    this.prompt.set('');

    if (this.mode() === 'stream') {
      // ── Streaming path ──────────────────────────────────────────────────
      this.chat.isStreaming.set(true);
      this.chat.addMessage('assistant', '');       // placeholder

      this.chat.streamMessage(
        text, mdl,
        (token) => this.chat.updateLastAssistantMessage(token),
        ()      => this.chat.isStreaming.set(false),
        (err)   => { this.errorMsg.set(err); this.chat.isStreaming.set(false); },
      );

    } else if (this.mode() === 'multi') {
      // ── Multi-turn path ─────────────────────────────────────────────────
      this.chat.isLoading.set(true);
      this.chat.multiTurnChat(mdl).subscribe({
        next:  (res) => { this.chat.addMessage('assistant', res.response); this.chat.isLoading.set(false); },
        error: (err) => { this.errorMsg.set(err.message); this.chat.isLoading.set(false); },
      });

    } else {
      // ── Single-shot path ────────────────────────────────────────────────
      this.chat.isLoading.set(true);
      this.chat.sendMessage(text, mdl).subscribe({
        next:  (res) => { this.chat.addMessage('assistant', res.response); this.chat.isLoading.set(false); },
        error: (err) => { this.errorMsg.set(err.message); this.chat.isLoading.set(false); },
      });
    }
  }

  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      this.send();
    }
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    this.uploadName.set(file.name);
    this.uploadMsg.set('');
    this.isUploading.set(true);

    this.chat.uploadDocument(file).subscribe({
      next: () => {
        this.uploadMsg.set(`Indexed ${file.name}`);
        this.isUploading.set(false);
        input.value = '';
      },
      error: (err) => {
        this.uploadMsg.set(err.error?.detail ?? err.message ?? 'Upload failed');
        this.isUploading.set(false);
        input.value = '';
      },
    });
  }

  trackByIndex(index: number): number { return index; }
}
