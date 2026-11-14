# QA: megalodon

**FAILED** - score 76% of the targets met

| check | blocks upload | ok | target | what was measured |
|---|---|---|---|---|
| match | yes | **no** |  | 432 shots checked against their line: 0 another animal than the voice names (blocks above 2); 19 an animal under a line that names none (reported): [(23.3, 'shark'), (25.4, 'shark'), (27.6, 'shark'), (29.7, 'shark'), (44.5, 'shark'), (48.3, 'shark')]; 386 without an answer |
| named_animal | yes | yes |  | an animal other than the line names at none s |
| text | yes | yes |  | thumbnail thumbnail_e5.jpg: text 'DER KÖNIG', German correct True, readable True |
| sound | yes | yes |  | silence of 3 s or more at none s; music over the voice in minute(s) none; true peak -1.4 dBTP (max -0.5); loudness -14.0 LUFS (-17..-11) |
| duration | yes | yes |  | 26.4 min (22-27) |
| license | yes | yes |  | sources without a confirmed licence: none |
| frames | yes | yes |  | 316 frames looked at (one every 5 s): black at none s; decoding errors: none |
| hook | no | **no** | the strict rule in its detail | 16 shots in 30 s, 14 moving animal clips (need 3), not moving or no animal at [19.2, 21.1], cuts outside 1.5-3 s: none, sound hit: yes |
| ltx | no | yes | the strict rule in its detail | 13 LTX-Video clips (need 8) |
| subject | no | yes | landscapes <= 25% | animal shown where told: {'otodus_megalodon': 0.81, 'great_white_shark': 0.72, 'ancient_whale': 0.63, 'whales': 0.48, 'mako_shark': 0.0}; another animal than the line names at none s; landscapes 13% (max 15%); prehistoric animals in motion: 23 shots |
| pace | no | yes | average shot <= 7.0 s | average shot 3.6 s (3-5), hook 2.0 s (1.5-3); shown more than twice: {'otodus_megalodon_scene_35__s1': 3, 'otodus_megalodon_extra_00__s0': 3, 'otodus_megalodon_extra_05__s0': 3, 'otodus_megalodon_extra_06__s0': 3, 'otodus_megalodon_extra_01__s0': 3, 'otodus_megalodon_pose_0__s1': 3, 'standby_ancient_whale_00__s0': 3, 'standby_ancient_whale_00__s1': 3, 'otodus_megalodon_scene_34__s0': 3}; again within 3 min: ['standby_ancient_whale_00__s0'] |
| music | no | **no** | the strict rule in its detail | music under the voice per minute: [22.78, 22.86, 22.73, 22.93, 23.04, 22.87, 22.53, 22.57, 22.24, 22.35, 22.58, 22.61, 22.51, 22.89, 22.68, 22.52, 22.85, 22.73, 22.24, 23.19, 23.05, 22.8, 22.57, 22.51, 22.84, 23.13, 23.39] dB; music as loud as the voice in minute(s): none; music silent 4 s |
| background | no | yes | the strict rule in its detail | music + atmosphere under the voice per minute: [22.05, 21.98, 22.18, 21.97, 22.05, 22.0, 22.05, 21.95, 22.01, 22.11, 21.98, 22.0, 22.14, 22.0, 21.99, 22.22, 21.97, 22.01, 22.07, 22.08, 21.94, 22.22, 21.99, 22.06, 22.0, 22.02, 22.0] dB (need >= 20); minutes too loud: none |
| ambience | no | yes | the strict rule in its detail | 109 atmosphere stretches, only inside their shots; sea under a picture without sea at none s |
| first_min | no | yes | the strict rule in its detail | 72% moving in the first minute (need 70 %) |
| motion | no | yes | animals in motion >= 25% | prehistoric animals in motion 50% of the film (need 40%): LTX 72 s, real footage of the animals 34 s, 2.5D parallax 690 s |
| dinosaurs | no | **no** | the strict rule in its detail | 0 dinosaurs in motion (need 2): LTX clips none, FLUX dinosaurs moved by the 2.5D parallax none |
| wow | no | yes | the strict rule in its detail | 122 wow scenes; the longest stretch without one 67 s (reported; the rule is a scene in every chapter, see scenes) |
| diversity | no | yes | the strict rule in its detail | main animal (great_white_shark) 14% of the film (max 40%); kinds: {'other_animal': '62%', 'main': '14%', 'environment': '12%', 'people': '6%', 'other': '4%', 'map': '3%'}; chapters with fewer than 3 kinds: none |
| scenes | no | yes | the strict rule in its detail | 122 dramatic animated scenes (LTX or real film); chapters without one: none |
| repeats | no | **no** | the strict rule in its detail | 148 sources for 443 shots; more than 2 times: {'clip:ltx-otodus_megalodon_pose_0__s0': 8, 'clip:ltx-standby_great_white_shark_00__s1': 6, 'clip:ltx-otodus_megalodon_scene_34__s1': 7, 'clip:ltx-otodus_megalodon_scene_36__s0': 7, 'clip:ltx-standby_great_white_shark_02__s0': 6, 'clip:ltx-standby_great_white_shark_03__s0': 6, 'clip:ltx-standby_great_white_shark_04__s0': 6, 'clip:ltx-otodus_megalodon_scene_37__s1': 8}; again within 3 min: {'clip:ltx-otodus_megalodon_pose_0__s0': [0, 44, 363, 437, 707, 912, 1082, 1297], 'clip:ltx-standby_great_white_shark_00__s1': [2, 162, 204, 226, 328, 1442], 'clip:ltx-otodus_megalodon_scene_34__s1': [4, 48, 367, 710, 916, 1086, 1300], 'clip:ltx-otodus_megalodon_scene_36__s0': [9, 100, 415, 757, 937, 1169, 1356], 'clip:ltx-standby_great_white_shark_02__s0': [10, 165, 208, 230, 339, 1446], 'clip:ltx-standby_great_white_shark_03__s0': [14, 168, 212, 233, 346, 1489]} |
| consistent | no | yes | the strict rule in its detail | 4 animals compared across the film; another look for: none |
| viewer | no | yes | the strict rule in its detail | 0 places a viewer would switch off (max 3): none |
| thumbnail | no | yes | the strict rule in its detail | chosen: thumbnail_e5.jpg; judge: {'animal': 'Megalodon', 'animal_big_and_recognisable': True, 'prehistoric_or_living_fossil': True, 'bright': True, 'words_readable': True, 'text': 'DER KÖNIG', 'german_correct': True, 'action_or_trick': True, 'forbidden_seen': [], 'pass': True, 'why': 'Das Vorschaubild zeigt einen riesigen, sofort erkennbaren Haikopf mit offenem Maul in starkem Kontrast, hat einen prägnanten und grammatikalisch korrekten zweiteiligen Text ("DER KÖNIG") und erfüllt alle Kriterien der Nische.'} |

- contact sheet: [contact.jpg](contact.jpg) (316 frames, one every 5 s)
- voice and music: [sound.png](sound.png)
- thumbnails: [thumb_1](thumb_1.jpg), [thumb_2](thumb_2.jpg), [thumb_3](thumb_3.jpg)
