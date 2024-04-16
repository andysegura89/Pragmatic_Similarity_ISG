import { Component } from '@angular/core';
import { Clip } from '../../Clip';
import { NgFor } from '@angular/common';
import { ClipsService } from '../../services/clips.service';
import { ClipItemComponent } from '../clip-item/clip-item.component';
import { ButtonComponent } from '../button/button.component';
import { Subscription } from 'rxjs';
@Component({
  selector: 'app-clips',
  standalone: true,
  imports: [NgFor, ClipItemComponent, ButtonComponent],
  templateUrl: './clips.component.html',
  styleUrl: './clips.component.css'
})
export class ClipsComponent {
  clips: Clip[] = [];
  subscription: Subscription | null = null;
  isStreaming: boolean = false;

  constructor(private clipsService: ClipsService) {}

  ngOnInit(): void{
  }

  toggleRecord(): void {
    if (this.isStreaming) {
      this.stopStream();
    } else {
      this.startStream();
    }
    this.isStreaming = !this.isStreaming;
  }

  startStream(): void{
    this.subscription = this.clipsService.getServerSentEvent('http://localhost:5050/stream/').subscribe((clip) => {
      this.clips.push(JSON.parse(clip['data']));
    });
  }

  stopStream(): void {
    this.subscription.unsubscribe();
    this.subscription = null;
    this.clipsService.stopRecording();
  }

  playClip(path: string) {
    this.clipsService.playClip(path);
  }

  deleteClip(clip: Clip) {
    console.log('deleting clip finally')
    this.clips = this.clips.filter(c => c !== clip)
  }
}
