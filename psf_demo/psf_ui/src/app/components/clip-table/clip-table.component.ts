import { Component } from '@angular/core';
import { Clip } from '../../Clip';
import { NgFor, NgIf } from '@angular/common';
import { ClipsService } from '../../services/clips.service';
import { ClipItemComponent } from '../clip-item/clip-item.component';
import { ButtonComponent } from '../button/button.component';
import { RecordingComponent } from '../recording/recording.component';
import { Subscription } from 'rxjs';
import { faTimes, faPlay } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { NgStyle } from '@angular/common';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';


@Component({
  selector: 'app-clip-table',
  standalone: true,
  imports: [NgFor, 
            NgIf,
            NgStyle, 
            ClipItemComponent, 
            ButtonComponent, 
            RecordingComponent, 
            FontAwesomeModule,
            MatProgressSpinnerModule],
  templateUrl: './clip-table.component.html',
  styleUrl: './clip-table.component.css'
})
export class ClipTableComponent {
  clips: Clip[] = [];
  pageSizes: Array<number> = [5, 10, 20];
  subscription: Subscription | null = null;

  isStreaming: boolean = false;
  currentPage: number = 1;
  pageSize: number = 5;
  faTimes = faTimes;
  faPlay = faPlay;

  constructor(private clipsService: ClipsService) {}

  ngOnInit(): void{
    this.visibleData();
    this.subscription = this.clipsService.getServerSentEvent('http://localhost:5050/stream/').subscribe((clip) => {
      var clip_actual = JSON.parse(clip['data']);
      if (clip_actual.best_path == null) {
        this.clips.push(JSON.parse(clip['data']));
      }
      else {
        for (var temp_clip of this.clips) {
          if (temp_clip.id == clip_actual.id) {
            temp_clip.best_cos = clip_actual.best_cos;
            temp_clip.best_path = clip_actual.best_path;
            temp_clip.best_asd_label = clip_actual.best_asd_label;
            temp_clip.second_cos = clip_actual.second_cos;
            temp_clip.second_path = clip_actual.second_path;
            temp_clip.second_asd_label = clip_actual.second_asd_label;
            temp_clip.hundred_path = clip_actual.hundred_path;
            temp_clip.hundred_cos = clip_actual.hundred_cos;
            temp_clip.hundred_asd_label = clip_actual.hundred_asd_label;
            temp_clip.five_hundred_path = clip_actual.five_hundred_path;
            temp_clip.five_hundred_cos = clip_actual.five_hundred_cos;
            temp_clip.five_hundred_asd_label = clip_actual.five_hundred_asd_label;
            break;
          }
        }
      }
    });

  }


  visibleData(): Clip[]{
    var startIdx = (this.currentPage - 1) * this.pageSize;
    var endIdx = startIdx + this.pageSize;
    return this.clips.slice(startIdx, endIdx)
  }

  nextPage(){
    if (this.currentPage < this.clips.length / this.pageSize) {
      this.currentPage += 1;
    } 
  }

  previousPage(){
    if (this.currentPage > 1) {
      this.currentPage -= 1;
    } 
  }

  changePage(pageNumber: number) {
    this.currentPage = pageNumber;
  }

  changePageSize(pageSize: any) {
    this.pageSize = pageSize;
  }

  pageNumbers(): Array<Object> {
    let totalPages = Math.ceil(this.clips.length / this.pageSize);
    let pageNumArray = new Array(totalPages);
    return pageNumArray;
  }

  toggleRecord(dataset: string): void {
    this.clipsService.toggleRecord(dataset);
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
    this.clips = this.clips.filter(c => c !== clip)
  }

  reverseList() {
    this.clips = this.clips.reverse();
  }

  loadSavedClips() {
    this.clipsService.loadSavedClips().subscribe(
      (data: Clip[]) => {
        this.clips = data;
      },
      error => {
        console.error('error in loading saved clips', error);
      }
    );
  }
}
