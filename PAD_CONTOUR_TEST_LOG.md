# Pad Contour Detection Test Log

Data: 2026-06-07

Scopul acestei etape este stabilizarea conturului verde pentru fiecare pad BGA
inainte de rafinarea detectiei de voiduri. In aceasta etapa nu se calibreaza
voidurile roz; se verifica doar daca ROI-ul/pad-ul verde este corect.

## Refactor V24 - ROI BGA + grila 16x16 pentru familia PCBA2

Data: 2026-06-09

Motiv:

- V21/V22 au reparat mult ROI-ul, dar `PCBA2_06` avea latura de jos prea jos
  in debug ROI si contururi haotice in interior.
- `PCBA2_04` ajungea la `256`, dar unele contururi erau prea mari sau se
  suprapuneau in zonele de fanout.
- Scopul V24 este sa pastram doar patratul BGA si sa obtinem `16 x 16 = 256`
  sloturi coerente pentru seria PCBA2.

Modificari importante:

1. ROI-ul BGA este decupat dupa rândurile dense de candidati, ca sa nu includa
   componente externe sub/supra BGA-ului.
2. Pentru imagini proiectate/ovalizate, se foloseste o grila cu pitch separat
   pe X si Y. Acest lucru este important pentru `PCBA2_06`, unde pad-urile nu
   apar ca cercuri perfecte.
3. Cand Hough devine instabil, se activeaza fallback pe componente intunecate
   locale in ROI: Otsu threshold + opening/closing + filtre de arie, extent,
   circularitate si aspect ratio.
4. Sloturile lipsa sunt recuperate doar daca au evidenta locala suficienta.
   Aceste pozitii sunt marcate intern ca `is_estimated=True`, deci nu sunt
   confundate cu masurari Hough directe.
5. Raza finala a contururilor este plafonata la `0.38 * pitch`, pentru a evita
   cercurile prea mari in zone dense precum `PCBA2_04`.

Fisier modificat:

- `src/ball_detection.py`

Script de debug adaugat:

- `temp_testare/generate_pcba2_v24_debug.py`

Output-uri V24:

- `temp_testare/pcba2_bga_roi_v24/01_raw_candidates`
- `temp_testare/pcba2_bga_roi_v24/02_detected_bga_roi`
- `temp_testare/pcba2_bga_roi_v24/03_filtered_real_pads`
- `temp_testare/pcba2_bga_roi_v24/04_rejected_candidates`
- `temp_testare/pcba2_bga_roi_v24/05_green_contours_only`
- `temp_testare/pcba2_bga_roi_v24/pcba2_v24_debug_summary.csv`

Rezumat PCBA2 dupa V24:

| Imagine | Pad-uri returnate | Candidati raw | Sloturi lipsa | Candidati respinsi | ROI |
| --- | ---: | ---: | ---: | ---: | --- |
| `PCBA2_03.jpg` | 256 | 439 | 0 | 183 | `11,763,2453,2486` |
| `PCBA2_03_.jpg` | 256 | 535 | 0 | 279 | `30,765,2435,2475` |
| `PCBA2_04.jpg` | 256 | 684 | 0 | 473 | `705,215,2309,1848` |
| `PCBA2_05.jpg` | 256 | 379 | 0 | 123 | `270,33,2715,2483` |
| `PCBA2_06.jpg` | 256 | 770 | 0 | 531 | `566,1141,2043,2079` |

Observatii:

- `PCBA2_06`: ROI-ul ramane taiat corect jos (`y_max=2079`) si nu mai coboara
  sub BGA. Grila finala are `256/256`; 35 pozitii sunt recuperate prin evidenta
  locala de grila.
- `PCBA2_04`: ramane `256/256`, dar contururile sunt mai stranse fata de V23
  dupa plafonarea razei la pitch.
- `PCBA2_03_`: nu mai pierde zona din dreapta sus; ajunge la `256/256`.

Verificari:

- `ruff check main.py src`: trecut.
- `main.py --image Images_BGA\Raw\PCBA2\PCBA2_04.jpg`: `detected 256 BGA solder balls`.
- `main.py --image Images_BGA\Raw\PCBA2\PCBA2_06.jpg`: `detected 256 BGA solder balls`.

## Imagine principala de calibrare

- Imagine: `Images_BGA/Raw/PCBA2/PCBA2_05.jpg`
- Baseline anterior: V14
- Versiune curenta testata: V15

## Probleme observate in V14

1. Unele pad-uri erau detectate doar pe zona centrala, nu pe tot pad-ul.
   Exemple verificate: zona centrala/fanout, in jurul coordonatelor:
   - `x1721 y1181`
   - `x1570 y1030`
   - `x1420 y1485`

2. Exista o detectie falsa pe trasee, fara pad real.
   Exemplu verificat:
   - `x255 y1633`, raza V14 aproximativ `36 px`

3. Hough Circle Transform era prea sensibil la structuri circulare locale:
   putea prinde miezul/voidul intern sau un model de trasee, nu limita reala
   a pad-ului.

## Modificari importante in V15

Fisier modificat:

- `src/ball_detection.py`

Modificarile principale:

1. Filtru pentru componente intunecate de tip traseu
   - Se analizeaza componenta intunecata conectata din jurul centrului cercului.
   - Daca aceasta este prea alungita si are aria prea mica fata de cerc,
     candidatul este respins.
   - Scop: eliminarea cercurilor verzi desenate peste trasee fara pad.

2. Normalizare contextuala a razei
   - Dupa filtrarea Hough si filtrarea de layout, se calculeaza raza mediana a
     pad-urilor detectate in imagine.
   - Candidatii cu raza prea mica sunt ridicati pana la raza mediana, cu un
     boost maxim controlat.
   - Scop: cazurile in care Hough detecteaza doar miezul pad-ului sunt extinse
     catre dimensiunea reala a pad-ului.

3. Rafinarea de margine ramane activa
   - Rafinarea radial-gradient introdusa anterior ramane activa, dar cu plafon
     adaptiv ca sa nu fie pacalita de linii negre.

## Teste rulate

### Test static

Comanda:

```powershell
.\.venv\Scripts\ruff.exe check main.py src
```

Rezultat:

- `All checks passed!`

### Test numeric V15 pe PCBA2_05

Comanda:

```powershell
.\.venv\Scripts\python.exe main.py --image Images_BGA\Raw\PCBA2\PCBA2_05.jpg --output-root temp_testare\contur_v15
```

Rezultat:

- Pad-uri detectate: `252`
- V14 avea: `253`
- Diferenta intentionata: falsul de pe trasee de la aproximativ `x255 y1633`
  a fost eliminat.

### Distributie raze V15

- Raza minima V15: `41 px`
- Raza maxima V15: `48 px`
- V14 avea raza minima mult mai mica in unele cazuri (`31 px`), ceea ce
  confirma problema "doar mijlocul pad-ului".

### Verificare fals eliminat

Coordonate verificate:

- `x255 y1633`

Rezultat V15:

- Nu mai exista detectie in jurul acestei zone.

Crop generat:

- `temp_testare/contur_v15_zones/false_removed_x110_430_y1500_1780.jpg`

### Verificare pad-uri cu raza mica corectata

Coordonate/zone verificate:

- `x1721 y1181`
- `x1570 y1030`
- `x1420 y1485`

Rezultat:

- Aceste pad-uri nu mai raman cu contur doar pe miez; raza minima a urcat
  contextual la aproximativ `41-42 px`.

Crop generat:

- `temp_testare/contur_v15_zones/small_radius_fixed_x1500_1840_y1000_1260.jpg`

### Sanity check pe folderul PCBA2

Rezultate detectie:

| Imagine | Pad-uri | Raza minima | Raza maxima |
| --- | ---: | ---: | ---: |
| `PCBA2_03.jpg` | 232 | 44 | 48 |
| `PCBA2_03_.jpg` | 211 | 41 | 47 |
| `PCBA2_04.jpg` | 44 | 32 | 45 |
| `PCBA2_05.jpg` | 252 | 41 | 48 |
| `PCBA2_06.jpg` | 29 | 39 | 48 |

Acest test este doar de sanitate, nu validare finala. Imaginile pot avea crop,
scara sau continut diferit.

## Output-uri V15 generate

Imagine full, doar contururi verzi:

- `temp_testare/contur_v15/PCBA2_05_green_contours_only.jpg`

Output pipeline:

- `temp_testare/contur_v15/Processed/PCBA2_05_03_annotated.jpg`
- `temp_testare/contur_v15/Processed/PCBA2_05_04_void_preview.jpg`

Crop-uri de verificare:

- `temp_testare/contur_v15_zones/full_top_row_x300_2200_y0_320.jpg`
- `temp_testare/contur_v15_zones/left_upper_x0_1050_y0_900.jpg`
- `temp_testare/contur_v15_zones/fanout_center_x650_2250_y600_1550.jpg`
- `temp_testare/contur_v15_zones/central_package_x900_2050_y650_1450.jpg`
- `temp_testare/contur_v15_zones/right_mid_x1850_3008_y700_1650.jpg`
- `temp_testare/contur_v15_zones/bottom_x250_2250_y1650_2512.jpg`
- `temp_testare/contur_v15_zones/right_bottom_x1850_3008_y1450_2400.jpg`
- `temp_testare/contur_v15_zones/false_removed_x110_430_y1500_1780.jpg`
- `temp_testare/contur_v15_zones/small_radius_fixed_x1500_1840_y1000_1260.jpg`

## Status V15

V15 este mai bun decat V14 pentru contururi:

- elimina falsul clar de pe trasee;
- ridica pad-urile detectate doar pe miez catre dimensiunea reala a pad-ului;
- pastreaza detectia pe `PCBA2_05` la aproape acelasi nivel numeric.

Ramane de verificat vizual de catre utilizator:

- daca toate cercurile verzi din marginile imaginii sunt relevante pentru setul
  de pad-uri BGA dorit;
- daca pad-urile din zona fanout sunt suficient de largi fara sa includa prea
  mult fundal;
- abia dupa validarea contururilor se va reveni la detectia de voiduri.

## Retouch V16

Motiv:

- In V15 mai existau cateva pad-uri reale fara contur verde, mai ales in partea
  dreapta si in zona inferioara dreapta.
- Acestea nu erau recuperabile sigur doar prin Hough relaxat, deoarece Hough
  relaxat producea sute de cercuri false in aceeasi regiune.

Strategie:

1. Nu s-a relaxat global Hough Circle Transform.
2. S-a adaugat recuperare controlata de goluri in grila:
   - se estimeaza pitch-ul dominant al BGA-ului din randurile/coloanele deja
     detectate;
   - se cauta goluri compatibile cu pitch-ul intre vecini validati;
   - se accepta un nou cerc doar daca zona locala arata ca un pad intunecat,
     cu forma compacta si suficienta masa intunecata.

Fisier modificat:

- `src/ball_detection.py`

Functii/adaugiri importante:

- `recover_grid_gaps`
- `_recover_grid_gap_circles`
- `_estimate_grid_pitch`
- `_is_grid_gap_pad_candidate`

Rezultat pe `PCBA2_05.jpg`:

- V15: `252` pad-uri detectate
- V16: `256` pad-uri detectate

Pad-uri recuperate de V16:

| Coordonata aproximativa | Observatie |
| --- | --- |
| `x2622 y1026` | pad scapat in zona dreapta sus |
| `x2475 y1482` | pad scapat in zona dreapta/mijloc |
| `x1420 y1636` | pad acoperit de linii, completare de grila |
| `x2481 y2088` | pad scapat in zona inferioara dreapta |

Fals verificat:

- Zona veche falsa `x255 y1633` ramane fara contur verde.

Zona marcata cu X:

- Nu s-a fortat detectia pe obiectul intunecat ambiguu din afara grilei.
- Regula V16 prefera pad-urile care au suport de grila si aspect local compact.
- Aceasta zona ramane de validat vizual, deoarece poate contine artefacte sau
  structuri suprapuse care nu trebuie tratate automat ca BGA pad.

Output-uri V16:

- `temp_testare/contur_v16/PCBA2_05_green_contours_only.jpg`
- `temp_testare/contur_v16/PCBA2_05_recovered_highlighted.jpg`
- `temp_testare/contur_v16_zones_clean/right_mid_x1850_3008_y700_1650.jpg`
- `temp_testare/contur_v16_zones_clean/right_bottom_x1850_3008_y1450_2400.jpg`
- `temp_testare/contur_v16_zones_clean/weird_x_area_x2200_2700_y1800_2130.jpg`

Sanity check PCBA2 dupa V16:

| Imagine | Pad-uri | Raza minima | Raza maxima |
| --- | ---: | ---: | ---: |
| `PCBA2_03.jpg` | 235 | 44 | 48 |
| `PCBA2_03_.jpg` | 224 | 41 | 47 |
| `PCBA2_04.jpg` | 74 | 32 | 45 |
| `PCBA2_05.jpg` | 256 | 41 | 48 |
| `PCBA2_06.jpg` | 29 | 39 | 48 |

Nota: cresterea pe alte imagini trebuie inspectata separat. Pentru `PCBA2_05`,
V16 este momentan mai aproape de cerinta: cat mai multe pad-uri reale cu contur
verde, fara reintroducerea falsului clar de pe trasee.

## Retouch V17

Motiv:

- V16 detecta toate cele `16 x 16 = 256` pad-uri pe `PCBA2_05`, dar unele
  contururi erau inca usor prea conservatoare si ramaneau spre interiorul
  pad-ului.
- Pentru detectia ulterioara de voiduri, ROI-ul trebuie sa acopere cat mai bine
  pad-ul complet, nu doar zona centrala intunecata.

Strategie:

1. Nu s-a schimbat logica de detectie a centrelor si nici recuperarea de grila
   din V16.
2. S-a adaugat o normalizare finala a razei conturului:
   - se calculeaza o raza robusta din distributia pad-urilor detectate in
     imagine;
   - se foloseste percentila 90 a razelor ca floor pentru contururile ramase
     prea mici;
   - floor-ul este plafonat in raport cu pitch-ul estimat al grilei, pentru ca
     regula sa ramana adaptabila la alte PCB-uri/scari;
   - fiecare cerc poate fi marit doar cu un boost maxim controlat.

Fisier modificat:

- `src/ball_detection.py`

Functii/adaugiri importante:

- `normalize_contour_radii`
- `_normalize_contour_radii`
- `contour_radius_floor_percentile`
- `contour_radius_max_pitch_ratio`

Rezultat pe `PCBA2_05.jpg`:

- V16: `256` pad-uri detectate
- V17: `256` pad-uri detectate
- Raza minima V16: `41 px`
- Raza minima V17: `46 px`
- Raza maxima V17: `48 px`

Output-uri V17:

- `temp_testare/contur_v17/PCBA2_05_green_contours_only.jpg`
- `temp_testare/contur_v17/Processed/PCBA2_05_03_annotated.jpg`
- `temp_testare/contur_v17_zones/right_mid_x1850_3008_y700_1650.jpg`
- `temp_testare/contur_v17_zones/user_example_x1200_2100_y150_850.jpg`
- `temp_testare/contur_v17_zones/weird_x_area_x2200_2700_y1800_2130.jpg`
- `temp_testare/contur_v17_compare/right_mid_x1850_3008_y700_1650_v16_vs_v17.jpg`
- `temp_testare/contur_v17_compare/fanout_center_x650_2250_y600_1550_v16_vs_v17.jpg`

Sanity check PCBA2 dupa V17:

| Imagine | Pad-uri | Raza minima | Raza maxima |
| --- | ---: | ---: | ---: |
| `PCBA2_03.jpg` | 235 | 45 | 48 |
| `PCBA2_03_.jpg` | 224 | 45 | 47 |
| `PCBA2_04.jpg` | 74 | 33 | 45 |
| `PCBA2_05.jpg` | 256 | 46 | 48 |
| `PCBA2_06.jpg` | 29 | 39 | 48 |

Status V17:

- Pastreaza completitudinea V16 pe `PCBA2_05`: `256/256`.
- Contururile sunt usor mai largi si mai uniforme, mai potrivite ca ROI pentru
  detectia de voiduri.
- Regula este bazata pe distributia si pitch-ul imaginii, nu pe o raza fixa
  hardcodata pentru `PCBA2_05`.

## Retouch V18

Motiv:

- Dupa V17, contururile acopereau bine pad-ul, dar unele centre erau inca
  influentate local de voiduri, linii negre sau contrastul intern al pad-ului.
- Pentru BGA-uri regulate, centrele reale trebuie sa cada pe o grila stabila
  (`16 x 16` in `PCBA2_05`, dar acelasi principiu se poate aplica si la
  `8 x 8`, `32 x 32` sau crop-uri partiale).

Strategie:

1. Se pastreaza detectia completa V16/V17.
2. Se estimeaza liniile de grila pe X si Y din centrele deja detectate.
3. Se valideaza grila:
   - numar minim de linii pe fiecare axa;
   - fill ratio minim;
   - pitch cu variatie mica;
   - dispersie mica a centrelor pe fiecare linie.
4. Centrele cercurilor sunt mutate la cea mai apropiata intersectie de grila,
   doar daca sunt suficient de aproape.
5. Raza ramane adaptiva si plafonata pe baza pitch-ului imaginii.

Fisier modificat:

- `src/ball_detection.py`

Functii/adaugiri importante:

- `regularize_grid_centers`
- `_regularize_grid_centers`
- `_regularized_axis_positions`
- `_nearest_grid_position`

Rezultat pe `PCBA2_05.jpg`:

- V18: `256` pad-uri detectate
- Grila detectata: `16` coloane x `16` randuri
- Fiecare coloana are `16` pad-uri
- Fiecare rand are `16` pad-uri
- Dispersia centrelor dupa regularizare pe aceeasi linie: `0 px`
- Raza minima V18: `46 px`
- Raza maxima V18: `48 px`

Output-uri V18:

- `temp_testare/contur_v18/PCBA2_05_green_contours_only.jpg`
- `temp_testare/contur_v18/Processed/PCBA2_05_03_annotated.jpg`
- `temp_testare/contur_v18_zones/user_example_x1200_2100_y150_850.jpg`
- `temp_testare/contur_v18_zones/fanout_center_x650_2250_y600_1550.jpg`
- `temp_testare/contur_v18_zones/weird_x_area_x2200_2700_y1800_2130.jpg`
- `temp_testare/contur_v18_compare/user_example_x1200_2100_y150_850_v17_vs_v18.jpg`
- `temp_testare/contur_v18_compare/right_mid_x1850_3008_y700_1650_v17_vs_v18.jpg`

Sanity check PCBA2 dupa V18:

| Imagine | Pad-uri | Raza minima | Raza maxima |
| --- | ---: | ---: | ---: |
| `PCBA2_03.jpg` | 235 | 45 | 48 |
| `PCBA2_03_.jpg` | 224 | 45 | 47 |
| `PCBA2_04.jpg` | 74 | 33 | 45 |
| `PCBA2_05.jpg` | 256 | 46 | 48 |
| `PCBA2_06.jpg` | 29 | 39 | 48 |

Nota pentru ROI:

- Nu este ideal sa strangem cercul fix pe pixelul de margine vizibila, deoarece
  marginea pad-ului poate fi neclara in X-ray si poate fi acoperita de trasee.
- Pentru void detection este mai sigur ca ROI-ul verde sa includa complet pad-ul
  cu o marja mica, iar filtrarea de voiduri sa fie facuta ulterior in interiorul
  zonei de interes.

## Validare V18 pe toate imaginile Raw

Motiv:

- Inainte de revenirea la void detection, V18 a fost rulat pe toate imaginile
  gasite in `Images_BGA/Raw`, ca sa vedem cat de general se comporta detectorul
  de pad-uri pe alte PCB-uri/crop-uri.
- Scopul acestei rulari este validare vizuala, nu schimbare de praguri pentru
  fiecare poza in parte.

Output-uri generate:

- `temp_testare/validare_contururi_v18_all_raw/contours_only`
- `temp_testare/validare_contururi_v18_all_raw/annotated_with_ids`
- `temp_testare/validare_contururi_v18_all_raw/v18_contour_validation_summary.csv`
- `temp_testare/validare_contururi_v18_all_raw/v18_all_raw_contact_sheet.jpg`

Rezumat:

| Imagine | Pad-uri V18 | Raza minima | Raza maxima | Raza mediana |
| --- | ---: | ---: | ---: | ---: |
| `FastScan/PCBA-1_BGA_15um_fastscan_View 2.jpg` | 2 | 34 | 41 | 37.5 |
| `FastScan/PCBA-1_BGA_15um_fastscan_View 3.jpg` | 16 | 32 | 41 | 37.0 |
| `FastScan/PCBA-1_BGA_15um_fastscan_View 7.jpg` | 7 | 31 | 45 | 38.0 |
| `PCBA1/PCBA1_07.jpg` | 172 | 34 | 46 | 34.0 |
| `PCBA1/PCBA1_08.jpg` | 57 | 38 | 47 | 41.0 |
| `PCBA1/PCBA1_10.jpg` | 262 | 46 | 48 | 46.0 |
| `PCBA2/PCBA2_03.jpg` | 235 | 45 | 48 | 45.0 |
| `PCBA2/PCBA2_03_.jpg` | 224 | 45 | 47 | 45.0 |
| `PCBA2/PCBA2_04.jpg` | 74 | 33 | 45 | 33.0 |
| `PCBA2/PCBA2_05.jpg` | 256 | 46 | 48 | 47.0 |
| `PCBA2/PCBA2_06.jpg` | 29 | 39 | 48 | 41.0 |

Observatii pentru validarea vizuala:

- `PCBA2_05.jpg` ramane cazul de referinta: `256/256`, grila completa.
- Imaginile `PCBA2_03`, `PCBA2_03_`, `PCBA2_04`, `PCBA2_06` par crop-uri sau
  campuri partiale, deci numerele mai mici sunt asteptate.
- `PCBA1_10.jpg` are `262` detectii; trebuie verificat vizual deoarece poate
  include cateva structuri de margine/nerelevante pe langa pad-urile reale.
- `PCBA1_07.jpg` detecteaza multe pad-uri, dar trebuie verificata zona de
  margine pentru eventuale false positive.
- `FastScan` pare sa aiba contrast/polaritate diferita fata de setul principal
  PCBA. V18 detecteaza doar cateva structuri intunecate acolo; daca FastScan
  intra in metodologia finala, probabil merita un mod separat de detectie.

## Refactor V19 - BGA ROI + grila 16x16 pentru PCBA2

Motiv:

- V18 detecta bine `PCBA2_05`, dar cauta cercuri in toata imaginea.
- Pe `PCBA2_04` si `PCBA2_06`, acest lucru producea doua probleme:
  - erau detectate componente/pad-uri externe;
  - pad-urile reale din BGA erau ratate cand scara era mai mica.

Strategie V19:

1. Se detecteaza intai candidati circulari la scara redusa, doar pentru a gasi
   zona densa a BGA-ului.
2. Se estimeaza pitch-ul dominant al grilei.
3. Se cauta explicit cea mai coerenta grila `16 x 16`.
4. Se construieste ROI-ul BGA din aceasta grila.
5. Se pastreaza doar cele `256` pozitii ale grilei BGA.
6. Candidatii externi sunt respinsi in debug.
7. Pozitiile fara candidat Hough direct sunt pastrate ca pozitii estimate din
   grila, dar sunt marcate in diagnostics; nu sunt ascunse.

Fisier modificat:

- `src/ball_detection.py`

Adaugari importante:

- `detect_solder_balls_with_diagnostics`
- `BallDetectionDiagnostics`
- `MissingGridPosition`
- `_detect_bga_roi_candidates`
- `_fit_bga_grid`
- `_bga_pitch_candidates`
- `_build_bga_grid_balls`

Output-uri V19:

- `temp_testare/pcba2_bga_roi_v19/01_raw_candidates`
- `temp_testare/pcba2_bga_roi_v19/02_detected_bga_roi`
- `temp_testare/pcba2_bga_roi_v19/03_filtered_16x16_grid`
- `temp_testare/pcba2_bga_roi_v19/04_rejected_candidates`
- `temp_testare/pcba2_bga_roi_v19/05_green_contours_only`
- `temp_testare/pcba2_bga_roi_v19/pcba2_v19_debug_summary.csv`

Cod culori in `03_filtered_16x16_grid`:

- verde: pozitie confirmata de candidat circular local;
- galben: pozitie estimata din grila BGA, fara candidat Hough direct;
- rosu: pozitie slaba/neconfirmata, raportata explicit in diagnostics.

Rezumat PCBA2 dupa V19:

| Imagine | Pad-uri returnate | Candidati raw | Sloturi ocupate | Sloturi estimate | Sloturi slabe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `PCBA2_03.jpg` | 256 | 1941 | 246 | 10 | 0 |
| `PCBA2_03_.jpg` | 256 | 1061 | 209 | 47 | 13 |
| `PCBA2_04.jpg` | 256 | 1157 | 209 | 47 | 0 |
| `PCBA2_05.jpg` | 256 | 1914 | 256 | 0 | 0 |
| `PCBA2_06.jpg` | 256 | 1116 | 190 | 66 | 6 |

Verificari:

- `ruff check main.py src`: trecut.
- Rulare pipeline pe `PCBA2_06.jpg`: `detected 256 BGA solder balls`.

## Retus V21 - ROI rafinat si local-Hough mai complet in ROI

Motiv:

- V20 era corect ca principiu, dar `PCBA2_06.jpg` avea latura de sus a ROI-ului
  prea ridicata, prinsa de componente externe.
- `PCBA2_03.jpg` si `PCBA2_03_.jpg` mai aveau pad-uri reale ratate, mai ales in
  zone de margine/pad-uri alungite.
- Obiectivul V21 este sa creasca recall-ul fara sa revina la completare
  geometrica artificiala.

Strategie V21:

1. Local-Hough se ruleaza in ROI cand candidatii mari sunt sub `98%` din cele
   `256` pozitii asteptate.
2. Candidatii local-Hough validati intra in acelasi filtru explicabil: ROI,
   raza, scor local, vecini pe pitch si componenta dominanta.
3. ROI-ul de debug se rafineaza dupa componenta dominanta conectata pe pitch,
   nu dupa toate cercurile brute.
4. Cativa candidati de margine pot fi pastrati doar daca raman in ROI-ul
   componentei BGA dominante.
5. Scorul local are un bonus mic pentru void-uri luminoase compacte din interior,
   folosit doar ca indiciu auxiliar, nu ca regula ML.
6. Daca rezultatul ramane sub `256`, lipsurile raman neconfirmate; nu se deseneaza
   pozitii verzi inventate.

Fisier modificat:

- `src/ball_detection.py`

Script de debug adaugat:

- `temp_testare/generate_pcba2_v21_debug.py`

Output-uri V21:

- `temp_testare/pcba2_bga_roi_v21/01_raw_candidates`
- `temp_testare/pcba2_bga_roi_v21/02_detected_bga_roi`
- `temp_testare/pcba2_bga_roi_v21/03_filtered_real_pads`
- `temp_testare/pcba2_bga_roi_v21/04_rejected_candidates`
- `temp_testare/pcba2_bga_roi_v21/05_green_contours_only`
- `temp_testare/pcba2_bga_roi_v21/pcba2_v21_debug_summary.csv`

Rezumat PCBA2 dupa V21:

| Imagine | Pad-uri reale returnate | Candidati raw | Sloturi neconfirmate fata de 256 | Candidati respinsi | ROI |
| --- | ---: | ---: | ---: | ---: | --- |
| `PCBA2_03.jpg` | 256 | 439 | 0 | 183 | `11,763,2453,2486` |
| `PCBA2_03_.jpg` | 247 | 353 | 9 | 105 | `30,765,2449,2480` |
| `PCBA2_04.jpg` | 211 | 684 | 45 | 473 | `705,223,2309,1830` |
| `PCBA2_05.jpg` | 256 | 379 | 0 | 123 | `270,33,2715,2483` |
| `PCBA2_06.jpg` | 246 | 720 | 10 | 474 | `566,1104,2043,2361` |

Observatii:

- `PCBA2_06.jpg`: ROI-ul nu mai porneste de la `y=844` ca in V20, ci de la
  `y=1104`, deci latura de sus este mai aproape de patratul BGA real.
- `PCBA2_03.jpg`: ajunge la `256/256` fara completare geometrica artificiala.
- `PCBA2_03_.jpg`: creste fata de V20, dar ramane conservator cu `247/256`
  confirmate si `9` pozitii neconfirmate.
- `PCBA2_04.jpg`: ramane cazul dificil; sunt confirmate `211` pad-uri, iar
  `45` pozitii raman neconfirmate pentru retus separat.
- `PCBA2_06.jpg`: nu este fortat la `256` daca fortarea introduce componente
  externe; raman `10` pozitii neconfirmate.

Verificari:

- `ruff check main.py src`: trecut.
- Rulare pipeline pe `PCBA2_06.jpg`: `detected 246 BGA solder balls`.

Nota:

- V19 rezolva problema principala pentru seria PCBA2: nu mai pastreaza cercuri
  externe ca rezultat final, ci doar grila BGA `16 x 16`.
- Pentru etapele de void detection, sloturile galbene/rosii trebuie tratate ca
  pozitii cu incredere mai mica, mai ales cand calculam procente si raportari.

## Refactor V20 - ROI BGA inainte de detectia pad-urilor reale

Motiv:

- V19 a fost respins la inspectia vizuala: returna `256` pozitii, dar unele erau
  pozitii estimate/geometrice si nu pad-uri reale masurate.
- Scopul V20 este sa detecteze intai patratul/ROI-ul BGA, apoi sa pastreze doar
  candidati observati direct in imagine.
- Daca nu se confirma toate cele `256` pozitii, lipsurile sunt raportate in CSV
  ca sloturi neconfirmate; nu mai sunt desenate ca pad-uri verzi.

Strategie V20:

1. Se aplica in continuare CLAHE + median filter in etapa de preprocessing.
2. Se extrag candidati de pad prin componente intunecate compacte
   (threshold Otsu + opening/closing + contour/connected-component metrics).
3. ROI-ul BGA este estimat din componenta densa cu pitch regulat, folosind
   raza candidatilor mari ca sa nu confundam vias-urile mici cu pad-urile.
4. Hough Circle Transform ramane in flux, dar nu mai are voie sa completeze
   liber grila; este folosit conservator si doar langa candidati mari existenti.
5. Se filtreaza candidatii dupa ROI, raza, scor local, vecini la pitch BGA si
   componenta principala conectata prin pitch.
6. Daca apar mai mult de `256` candidati, se pastreaza cei mai consistenti cu
   vecinatatea/grila; daca apar mai putini, nu se inventeaza pozitii.

Fisier modificat:

- `src/ball_detection.py`

Script de debug adaugat:

- `temp_testare/generate_pcba2_v20_debug.py`

Output-uri V20:

- `temp_testare/pcba2_bga_roi_v20/01_raw_candidates`
- `temp_testare/pcba2_bga_roi_v20/02_detected_bga_roi`
- `temp_testare/pcba2_bga_roi_v20/03_filtered_real_pads`
- `temp_testare/pcba2_bga_roi_v20/04_rejected_candidates`
- `temp_testare/pcba2_bga_roi_v20/05_green_contours_only`
- `temp_testare/pcba2_bga_roi_v20/pcba2_v20_debug_summary.csv`

Rezumat PCBA2 dupa V20:

| Imagine | Pad-uri reale returnate | Candidati raw | Sloturi neconfirmate fata de 256 | Candidati respinsi | ROI |
| --- | ---: | ---: | ---: | ---: | --- |
| `PCBA2_03.jpg` | 246 | 412 | 10 | 166 | `0,636,2410,2613` |
| `PCBA2_03_.jpg` | 244 | 349 | 12 | 105 | `0,704,2488,2542` |
| `PCBA2_04.jpg` | 205 | 340 | 51 | 133 | `592,234,2290,1932` |
| `PCBA2_05.jpg` | 256 | 379 | 0 | 123 | `314,77,2671,2439` |
| `PCBA2_06.jpg` | 180 | 400 | 76 | 218 | `569,844,2057,2331` |

Observatii:

- `PCBA2_05.jpg` ramane cazul complet: `256/256` confirmate.
- `PCBA2_03.jpg` si `PCBA2_03_.jpg` sunt aproape complete si nu mai sunt
  umplute artificial pana la `256`.
- `PCBA2_04.jpg` are ROI mult mai apropiat de patratul BGA, dar inca are multe
  pozitii neconfirmate; necesita urmator retus pe recover-ul pad-urilor reale.
- `PCBA2_06.jpg` nu mai pastreaza candidatele externe de componente ca rezultat
  final, dar detecteaza doar componenta principala BGA confirmata. Lipsurile
  ramase trebuie tratate explicit in urmatorul pas, nu completate geometric.

Verificari:

- `ruff check main.py src`: trecut.
