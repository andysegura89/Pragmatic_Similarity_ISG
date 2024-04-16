import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class SseService {

  constructor() { }

  getEventSource(url: string){
    return new EventSource(url);
  }
}
