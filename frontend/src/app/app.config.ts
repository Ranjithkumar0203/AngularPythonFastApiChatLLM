import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideRouter }                                 from '@angular/router';
import { provideHttpClient, withFetch }                  from '@angular/common/http';
import { provideAnimationsAsync }                        from '@angular/platform-browser/animations/async';
import { routes }                                        from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    // Angular 20: event coalescing reduces unnecessary CD cycles
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideRouter(routes),
    // withFetch() uses the Fetch API instead of XHR — Angular 20 default
    provideHttpClient(withFetch()),
    provideAnimationsAsync(),
  ],
};
