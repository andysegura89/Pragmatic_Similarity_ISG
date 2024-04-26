import { Component } from '@angular/core';
import { ButtonComponent } from '../button/button.component';
import { ClipsService } from '../../services/clips.service';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [ButtonComponent],
  templateUrl: './header.component.html',
  styleUrl: './header.component.css'
})
export class HeaderComponent {
  title: string = "Prosody Similiarity Finder";
  constructor(private clipsService: ClipsService) {
  }

  ngOnInit(): void {}

  toggleRecord() {
    console.log('toggle');
    this.clipsService.recordClips(true);
  }
}
