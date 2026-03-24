import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    // Lazy-load the chat feature (standalone component)
    loadComponent: () =>
      import('./chat/chat.component').then(m => m.ChatComponent),
  },
  { path: '**', redirectTo: '' },
];
