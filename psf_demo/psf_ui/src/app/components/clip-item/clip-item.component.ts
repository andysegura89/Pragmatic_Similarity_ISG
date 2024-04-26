import { Component, Input, Output, EventEmitter } from '@angular/core';
import { Clip } from '../../Clip';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { faTimes, faPlay } from '@fortawesome/free-solid-svg-icons';
import { NgStyle } from '@angular/common';

@Component({
  selector: 'app-clip-item',
  standalone: true,
  imports: [FontAwesomeModule, NgStyle],
  templateUrl: './clip-item.component.html',
  styleUrl: './clip-item.component.css'
})

export class ClipItemComponent {
  @Input() clip: Clip;
  @Input() color: string;
  @Output() onPlayClip: EventEmitter<string> = new EventEmitter();
  @Output() onDeleteClip: EventEmitter<Clip> = new EventEmitter();

  faTimes = faTimes;
  faPlay = faPlay;

  onPlay(path: string): void {
    this.onPlayClip.emit(path)
  }

  onDelete(clip: Clip) {
    console.log('inside clip item delete', clip);
    this.onDeleteClip.emit(clip);
  }
}
