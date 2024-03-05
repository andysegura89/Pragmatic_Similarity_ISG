import sys

english_winners = [0, 2, 41, 54, 63, 67, 102, 155, 159, 172, 173, 193, 197, 248,
                   261, 280, 286, 350, 358, 431, 459, 482, 488, 528, 603, 616,
                   618, 707, 708, 715, 717, 731, 792, 799, 804, 809, 824, 828,
                   870, 878, 880, 900, 903, 959, 45, 113, 120, 121, 204, 206,
                   246, 269, 411, 453, 510, 559, 602, 656, 666, 929, 972, 973,
                   1013, 1016, 13, 17, 105, 134, 136, 185, 188, 474, 578, 622,
                   651, 882, 925, 162, 410, 577, 628, 750, 758, 866, 869,
                   952, 963, 965, 50, 168, 436, 470, 513, 527, 557, 660,
                   732, 514, 661, 694, 698, 935, 937]

spanish_winners = [41, 48, 67, 85, 151, 226, 227, 261, 346, 347, 354, 357, 411,
                   415, 440, 448, 463, 468, 474, 510, 518, 528, 530, 585, 603,
                   609, 614, 618, 665, 703, 705, 732, 736, 792, 793, 814, 830,
                   835, 852, 858, 864, 895, 900, 906, 912, 915, 937, 971, 977,
                   1020, 1023, 174, 177, 206, 245, 249, 266, 397, 490, 494, 666,
                   689, 815, 861, 81, 208, 280, 398, 407, 409, 525, 533, 926, 927,
                   117, 287, 384, 655, 656, 809, 887, 897, 31, 36, 112, 231, 232,
                   472, 807, 881, 932, 944, 1004, 62, 63, 115, 155, 551, 682, 946, 1007]

"""
We implemented a custom greedy algorithm to find the features
more useful for detecting pragmatic similarity. We found 103 english features
and 101 spanish features. 
"""

def remove_losing_features(all_features, language='english'):
    if len(all_features) != 1024:
        print(f"features given must be of length 1024 not {len(all_features)}")
        sys.exit()
    selected_features = spanish_winners if language == 'spanish' else english_winners
    return [all_features[feature_idx] for feature_idx in selected_features]
