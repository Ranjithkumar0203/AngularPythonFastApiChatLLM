import { TestBed }       from '@angular/core/testing';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient }        from '@angular/common/http';
import { ChatComponent }  from './chat.component';
import { ChatService }    from './chat.service';

describe('ChatComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatComponent],
      providers: [provideHttpClient(), provideHttpClientTesting(), ChatService],
    }).compileComponents();
  });

  it('should create', () => {
    const fixture = TestBed.createComponent(ChatComponent);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('canSend should be false with empty prompt', () => {
    const fixture = TestBed.createComponent(ChatComponent);
    const comp = fixture.componentInstance;
    comp.prompt.set('');
    expect(comp.canSend()).toBeFalse();
  });

  it('canSend should be true with non-empty prompt', () => {
    const fixture = TestBed.createComponent(ChatComponent);
    const comp = fixture.componentInstance;
    comp.prompt.set('Hello');
    expect(comp.canSend()).toBeTrue();
  });
});
