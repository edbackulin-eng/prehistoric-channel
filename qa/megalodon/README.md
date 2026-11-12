# QA: megalodon

**FAILED**

| check | ok | what was measured |
|---|---|---|
| hook | **no** | 9 shots in 30 s, 9 moving animal clips (need 3), not moving or no animal at none, cuts outside 1.5-3 s: [16.53], sound hit: yes |
| ltx | yes | 9 LTX-Video clips (need 8) |
| subject | **no** | animal shown where told: {'great_white_shark': 0.76, 'otodus_megalodon': 0.0, 'ancient_whale': 0.68, 'whales': 0.6}; another animal than the line names at none s; landscapes 35% (max 15%); prehistoric animals in motion: 17 shots |
| pace | **no** | average shot 15.1 s (3-5), hook 3.6 s (1.5-3); shown more than twice: none; again within 3 min: none |
| music | **no** | music under the voice per minute: [23.13, 22.96, 22.92, 23.27, 23.31, 22.89, 23.33, 23.23, 23.08, 23.04, 23.0, 22.91, 22.74, 23.2, 22.97, 22.6, 23.36, 22.8, 22.97, 22.61, 22.9, 23.05, 23.49, 23.06, 22.78, 23.11, 23.48] dB; music as loud as the voice in minute(s): none; music silent 2 s |
| background | yes | music + atmosphere under the voice per minute: [22.0, 22.0, 22.15, 21.99, 22.0, 22.21, 21.97, 22.07, 21.99, 22.03, 22.01, 21.97, 22.23, 21.98, 22.0, 22.37, 21.91, 22.13, 21.96, 22.17, 21.98, 22.27, 21.96, 22.12, 22.08, 22.01, 22.0] dB (need >= 20); minutes too loud: none |
| ambience | yes | 44 atmosphere stretches, only inside their shots; sea under a picture without sea at none s |
| first_min | yes | 86% moving in the first minute (need 70 %) |
| motion | **no** | prehistoric animals in motion 37% of the film (need 40%): LTX 73 s, real footage of the animals 170 s, 2.5D parallax 333 s |
| dinosaurs | **no** | 1 dinosaurs in motion (need 2): LTX clips ['ltx-cretaceous_trex_stride__s1'], FLUX dinosaurs moved by the 2.5D parallax none |
| wow | yes | 1 wow scenes; the longest stretch without one 1153 s (reported; the rule is a scene in every chapter, see scenes) |
| diversity | yes | main animal (great_white_shark) 19% of the film (max 40%); kinds: {'environment': '34%', 'other_animal': '20%', 'main': '19%', 'other': '16%', 'map': '8%', 'people': '2%'}; chapters with fewer than 3 kinds: none |
| scenes | **no** | 1 dramatic animated scenes (LTX or real film); chapters without one: ['hook', 'story', 'finale'] |
| repeats | yes | 72 sources for 111 shots; more than 2 times: none; again within 3 min: none |
| match | **no** | 101 shots checked against their line, 6 wrong or unnamed animal (max 2): [(4.3, 'dinosaur'), (91.8, 'goldfish'), (206.8, 'crab'), (290.2, 'none'), (652.1, 'whale'), (1456.3, 'jellyfish')]; 0 without an answer |
| consistent | **no** | 3 animals compared across the film; another look for: {'great_white_shark': {'at': [414], 'why': 'Bild 8 zeigt einen Haifisch mit deutlich hellerem, bräunlichem Kopf- und Rückenfaktor sowie abweichenden Proportionen un'}, 'ancient_whale': {'at': [269, 528, 534], 'why': 'Die Bilder zeigen völlig unterschiedliche Tiere (z.B. gemusterter Urzeitwal bei 1/5, glatter grauer Kopf bei 2, und mode'}, 'whales': {'at': [988, 997], 'why': 'Die Bilder 4 und 6 zeigen einen echten Buckelwal, während die anderen Bilder ein CGI-Modell eines Urzeit-Wals (z.B. Basi'}} |
| viewer | **no** | 171 places a viewer would switch off (max 3): [('0:20', 'wiederholt', 'Fast exakt dasselbe Hai-Bild wie bei 0:15.'), ('0:25', 'wiederholt', 'Schon wieder genau dieselbe Hai-Perspektive.'), ('0:45', 'wiederholt', 'Identische Küstenszene wie im Bild davor.'), ('1:10', 'wiederholt', 'Exakt dieselbe Weltkarte mit Zoom auf Afrika wird '), ('1:20', 'wiederholt', 'Wiederholt exakt das Wolken- und Strandbild von 1:'), ('1:35', 'langweilig', 'Unruhige, wenig aussagekräftige Wasseraufnahme.'), ('1:45', 'wiederholt', 'Das blaue Unterwasserlicht wiederholt sich fast id'), ('1:50', 'wiederholt', 'Erneut derselbe blaue Strömungseffekt ohne neue In')] |
| thumbnail | yes | chosen: thumbnail_e8.jpg; judge: {'animal': 'Megalodon', 'animal_big_and_recognisable': True, 'prehistoric_or_living_fossil': True, 'bright': True, 'words_readable': True, 'text': 'DER KÖNIG', 'german_correct': True, 'action_or_trick': True, 'forbidden_seen': [], 'pass': True, 'why': 'Tier füllt das Bild aus, knallige Türkis-Farben, Text ist kurz und grammatikalisch korrekt, offenes Maul ist als Aktion vorhanden.'} |

- contact sheet: [contact.jpg](contact.jpg) (315 frames, one every 5 s)
- voice and music: [sound.png](sound.png)
- thumbnails: [thumb_1](thumb_1.jpg), [thumb_2](thumb_2.jpg), [thumb_3](thumb_3.jpg)
