import { Component, Output, EventEmitter } from '@angular/core';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { faMicrophone } from '@fortawesome/free-solid-svg-icons';
import { trigger, state, style, transition, animate } from '@angular/animations';
import { NgIf } from '@angular/common';
import {FormsModule} from '@angular/forms';
import {MatRadioModule} from '@angular/material/radio';
import {MatCheckboxModule} from '@angular/material/checkbox';


@Component({
  selector: 'app-recording',
  standalone: true,
  imports: [NgIf, FontAwesomeModule, FormsModule, MatRadioModule, MatCheckboxModule],
  templateUrl: './recording.component.html',
  styleUrl: './recording.component.css',
  animations: [
    trigger('recordingState', [
      state('inactive', style({ 
        transform: 'scale(1)',
        color: 'black'
      })),
      state('pulse', style({ 
        transform: 'scale(1)',
        color: 'red'
       })),
      transition('inactive <=> pulse', animate('.5s ease-in-out'))
    ]),
    trigger('recordingOutline', [
      state('inactive', style({
      })),
      state('recording', style({
        border: '2px solid red'
      })),
      transition('inactive <=> recording', animate('.5s ease-in-out'))
    ])
  ]
})
export class RecordingComponent {
  @Output() onToggleRecord: EventEmitter<string> = new EventEmitter();
  isRecording: boolean = false;
  faMicrophone = faMicrophone;
  selectedDataset: string ='DRAL';
  datasets: string[] = ['DRAL', 'SWBD', 'ASDNT'];
  male: boolean = true;
  female: boolean = true;
  
  toggleRecording(){
    this.isRecording = !this.isRecording;
    if (this.selectedDataset == 'SWBD') {
      var postfix = '-';
      if (this.male) {
        postfix += 'M';
      }
      if (this.female) {
        postfix += 'F';
      }
      if (!this.male && !this.female) {
        postfix += 'MF';
      }
      this.onToggleRecord.emit(this.selectedDataset + postfix)
    }
    else {
      this.onToggleRecord.emit(this.selectedDataset);
    }
    
  }
}
