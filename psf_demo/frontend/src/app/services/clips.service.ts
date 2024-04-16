import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Clip } from '../Clip';
import { NgZone } from '@angular/core';
import { SseService } from './sse.service';

@Injectable({
  providedIn: 'root'
})
export class ClipsService {
  private apiUrl = "http://localhost:5050/clips/";
  private stopRecordingUrl = "http://localhost:5050/stop_recording/";
  private playClipUrl = "http://localhost:5050/play_clip/";

  constructor(private http:HttpClient, private _zone:NgZone, private _sseService:SseService) { }

  getClips(): Observable<Clip[]> {
    return this.http.get<Clip[]>(this.apiUrl);
    const eventSource = new EventSource('http://localhost:5050/clips/');

  }

  recordClips(recording: boolean): void {
    console.log("inside record clips");
    const body = { recording: recording };
    this.http.put<any>(this.apiUrl, body)
    .subscribe(
      () => console.log('PUT request successful'),
      error => console.error('PUT request failed:', error)
    );
  }

  stopRecording(): void {
    const body = { recording: false };
    this.http.put<any>(this.stopRecordingUrl, JSON.stringify(body), 
    
    {headers: new HttpHeaders({
      'Access-Control-Allow-Origin': '*'
    })})
    .subscribe(
      () => console.log('PUT request successful'),
      error => console.error('PUT request failed:', error)
    );
  }

  getServerSentEvent(url: string){
    return Observable.create(observer => {
      const eventSouce = this._sseService.getEventSource(url);

      eventSouce.onmessage = event => {
        this._zone.run(() => {
          observer.next(event);
        });
      };

      eventSouce.onerror = error => {
        this._zone.run(() => {
          observer.error(error);
        })
      }
    })
  }

  playClip(path: string) {
    const body = { 'path': path };
    this.http.put<any>(this.playClipUrl + path, body).subscribe(
      () => console.log('PUT request successful'),
      error => console.error('PUT request failed:', error)
    );
  }
}
