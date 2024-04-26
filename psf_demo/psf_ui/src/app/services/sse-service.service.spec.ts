import { TestBed } from '@angular/core/testing';

import { SseServiceService } from './sse-service.service';

describe('SseServiceService', () => {
  let service: SseServiceService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SseServiceService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
