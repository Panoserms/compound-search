# Stage 7: Error Analysis of 50 Dev Queries

- **Split:** `dev` (1,000 train questions, seed 13)
- **Pool:** `own_hard_negatives_dropped` (headline pool)
- **Total sampled queries:** 50

## 1. Category Summary

| Category | Description | Total in Dev | Sampled |
|---|---|---:|---:|
| `splitting_wins` | Stemming + compound splitting rescues BM25 (bm25_parts_stem <= 10, bm25_plain > 10 & bm25_stem > 10) | 56 | 15 |
| `splitting_losses` | Compound splitting regression vs. stemming alone (bm25_stem <= 10, bm25_parts_stem > 10) | 28 | 10 |
| `hybrid_preprocessing_wins_h3` | H3 mechanism: preprocessed hybrid rescues plain hybrid (hybrid_parts_stem_dense <= 10, hybrid_plain_dense > 10) | 59 | 10 |
| `llm_rewrite_wins_h5` | H5 mechanism: Gemma-4 query expansion rescues hybrid (hybrid_parts_stem_dense_rewrite <= 10, hybrid_parts_stem_dense > 10) | 12 | 10 |
| `residual_hard_failures` | Hard residual failures (both hybrid_parts_stem_dense_rewrite > 10 and dense > 10) | 29 | 5 |

### Counts over all matching dev queries (not only the sampled ones)

| Category | n | question has cut word | gains shared term | loses shared term | plain bm25 outside top100 | best lexical ranks higher than plain | dense in top10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `splitting_wins` | 56 | 39 | 53 | 12 | 18 | 56 | 55 |
| `splitting_losses` | 28 | 16 | 12 | 12 | 4 | 7 | 23 |
| `hybrid_preprocessing_wins_h3` | 59 | 27 | 37 | 16 | 50 | 59 | 56 |
| `llm_rewrite_wins_h5` | 12 | 2 | 3 | 2 | 8 | 3 | 10 |
| `residual_hard_failures` | 29 | 11 | 11 | 9 | 8 | 10 | 0 |

## 2. Query-by-Query Breakdown (50 Queries)

### 1. [splitting_wins] Query `train-00445`
- **Question:** Welche Tierarten haben den besten Gesichtssinn? 
- **Gold Answer(s):** beispielsweise die Katzen und die Primaten | **Gold Article:** *Säugetiere* (`243cddb735c49a59`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=18, `parts_stem`=1, `dense`=1, `hyb_plain`=10, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** `tierarten` -> `tier + arten`, `gesichtssinn` -> `gesicht + sinn`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `art`, `gesicht`, `sinn`, `tier`
- **Gold Passage Snippet:** > === Sinneswahrnehmung === Der Geruchssinn spielt eine bedeutende Rolle in der Lebensweise der Säugetiere, unter anderem bei der Nahrungssuche und bei der Fortpflanzung, wo Pheromone die Paarungsbereitschaft signalisieren. Auch für das Territorialverhalten ist …

### 2. [splitting_wins] Query `train-00668`
- **Question:** Wie hoch ist die Einwohnerdichte in Melbourne?
- **Gold Answer(s):** 1978 Einwohner je Quadratkilometer | **Gold Article:** *Melbourne* (`b9522633b2ebf5c3`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=43, `stem`=50, `parts_stem`=1, `dense`=1, `hyb_plain`=10, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** `einwohnerdichte` -> `einwohner + dichte`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `dicht`, `einwohn`
- **Gold Passage Snippet:** > == Einwohnerentwicklung == Die Einwohnerzahl von Melbourne hat sich seit Ende der 1950er Jahre auf 3,4 Millionen im Jahr 2006 verdoppelt. Die Bevölkerungsdichte beträgt 1978 Einwohner je Quadratkilometer. In Berlin sind es zum Vergleich 3800. In den ländlichen…

### 3. [splitting_wins] Query `train-01625`
- **Question:** Warum konnte in Ostpreußen mit Wasser kaum Energie erzeugt werden? 
- **Gold Answer(s):** Das geringe Gefälle der Tieflandflüsse | **Gold Article:** *Ostpreußen* (`36287361ef326656`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=64, `parts_stem`=1, `dense`=1, `hyb_plain`=11, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** `ostpreußen` -> `ost + preußen`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `energi`, `ost`, `preuss`, `wass`
- **Gold Passage Snippet:** > === Industrie === Der Bernstein zählte zu den wenigen Bodenschätzen Ostpreußens, gab aber nur einigen tausend Menschen Arbeit. Er wurde im Tagebau bei Palmnicken gewonnen und in der Manufaktur in Königsberg verarbeitet. Das Fehlen von Steinkohle als Energieträ…

### 4. [splitting_wins] Query `train-02285`
- **Question:** Wie hoch ist das höchste Gebäude in Estland?
- **Gold Answer(s):** 314 Meter | **Gold Article:** *Estland* (`4c1046695cb4097c`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=12, `stem`=12, `parts_stem`=7, `dense`=1, `hyb_plain`=1, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** `estland` -> `est + land`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `est`, `land`
- **Gold Passage Snippet:** > === Architektur === Die estnischen Städte werden immer noch von den Holzhäusern geprägt, auch wenn die sowjetischen Plattenbauten dazwischen ragen. Heutzutage wird viel mit Schiefer gebaut. Das höchste Bauwerk Estlands ist der Fernsehturm in Tallinn (314 Meter…

### 5. [splitting_wins] Query `train-02441`
- **Question:** Wer macht Vorschläge für Gesetze in Mali?
- **Gold Answer(s):** Der Ministerrat | **Gold Article:** *Mali* (`6fbb143f2639e494`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=None, `parts_stem`=3, `dense`=3, `hyb_plain`=38, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `gesetz`, `vorschlag`
- **Gold Passage Snippet:** > === Politisches System === Das politische Leben Malis basiert auf der Verfassung. Sie wurde 1992 durch eine Volksabstimmung angenommen. Das republikanische Regierungssystem ist an dasjenige Frankreichs angelehnt und als semipräsidentiell zu bezeichnen. Es sieh…

### 6. [splitting_wins] Query `train-03502`
- **Question:** Wie viele Einwohner Irans leben im Armut?
- **Gold Answer(s):** zwischen 44,5 und 55 % der städtischen Bevölkerung | **Gold Article:** *Iran* (`8013b26b4560c07f`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=35, `parts_stem`=5, `dense`=2, `hyb_plain`=16, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `leb`
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Armut Iran Einwohner Armutsquote Iran Bevölkerung Armutsrate`
- **Gold Passage Snippet:** > == Sozialsystem == In seiner Schrift ''Der islamische Staat'' formulierte Ruhollah Chomeini die Verbesserung der Lebensumstände der armen Bevölkerung und die Beseitigung sozialer Ungleichheit als Ziele einer islamischen Gesellschaftsordnung:  93 % der iranisch…

### 7. [splitting_wins] Query `train-03951`
- **Question:** Nach wem ist die Musikhochschule in Weimar benannt?
- **Gold Answer(s):** Franz Liszt | **Gold Article:** *Thüringen* (`63ac9e083c694fa4`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=24, `stem`=32, `parts_stem`=2, `dense`=6, `hyb_plain`=7, `hyb_parts`=4, `hyb_rewrite`=4
- **Query Compound Splits:** `musikhochschule` -> `musik + hochschule`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `hochschul`, `musik`
- **Gold Passage Snippet:** > === Hochschulen === Einzige Volluniversität ist die 1558 gegründete Friedrich-Schiller-Universität Jena mit zehn Fakultäten, die gleichzeitig die älteste durchgehend bestehende Hochschule in Thüringen ist. Mit der Bauhaus-Universität in Weimar mit den Schwerpu…

### 8. [splitting_wins] Query `train-03970`
- **Question:** Welche Wirtschaftsbranchen auf der St.Helena Insel sind immer noch in der Entwicklungsphase?
- **Gold Answer(s):** Baugewerbe, Landwirtschaft, Handel, Immobilien und der Dienstleistungssektor | **Gold Article:** *St._Helena__Insel_* (`b4ae756350ace6b6`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=18, `stem`=20, `parts_stem`=1, `dense`=1, `hyb_plain`=3, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** `wirtschaftsbranchen` -> `wirtschaft + branchen`, `entwicklungsphase` -> `entwicklung + phase`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `wirtschaft`
- **Gold Passage Snippet:** > == Wirtschaft == Finanziell ist die Insel weitgehend auf die Unterstützung Großbritanniens angewiesen. Hauptindustriezweige sind der Fischfang und die Fischverarbeitung. Der Export von Fisch machte 60 % des Exportvolumens von 95.000 Pfund Sterling aus. Das Dur…

### 9. [splitting_wins] Query `train-05977`
- **Question:** In welchem Jahr kam Muammar al Gaddafi zur Welt?
- **Gold Answer(s):** 1942  | **Gold Article:** *Muammar_al-Gaddafi* (`eaa95fdf02733bbe`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=12, `stem`=12, `parts_stem`=9, `dense`=2, `hyb_plain`=5, `hyb_parts`=2, `hyb_rewrite`=2
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `welt`
- **Gold Passage Snippet:** > == Herkunft und Jugend == Muammar al-Gaddafi wurde nach späteren Angaben am 19. Juni 1942 in Sirte geboren. Andere Quellen geben als Geburtsort Qasr Abu Hadi bei Sirte an. Er entstammte einer Familie von Beduinen aus dem wenig einflussreichen Stamm der Guededf…

### 10. [splitting_wins] Query `train-06045`
- **Question:** Wer hat den obersten Befehl über die Kanadische Armee?
- **Gold Answer(s):** Königin Elisabeth II. | **Gold Article:** *Kanadische_Streitkräfte* (`585e701e68f8a722`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=None, `parts_stem`=3, `dense`=1, `hyb_plain`=17, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `befehl`
- **Gold Passage Snippet:** > === Befehlskette === Der Oberbefehlshaber der kanadischen Streitkräfte ist Königin Elisabeth II., Königin nach dem Recht Kanadas, vertreten durch den Generalgouverneur von Kanada, zur Zeit Julie Payette. Der militärische Führer ist der Chief of the Defence Sta…

### 11. [splitting_wins] Query `train-06059`
- **Question:** Was gab es bereits vor den Entwicklung von Getreideanbau?
- **Gold Answer(s):** eine jahrtausendelange Nutzung entsprechender Wildvorkommen | **Gold Article:** *Jungsteinzeit* (`e906e7463c2cc226`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=44, `stem`=57, `parts_stem`=2, `dense`=2, `hyb_plain`=3, `hyb_parts`=2, `hyb_rewrite`=2
- **Query Compound Splits:** `getreideanbau` -> `getreide + anbau`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `anbau`, `getreid`
- **Gold Passage Snippet:** > == Neolithische Revolution == In der Levante entstanden einige dauerhafte Siedlungen bereits ''vor'' der Entwicklung der Landwirtschaft. Die Umgebung dieser Siedlungen bot den Bewohnern in der Allerödzeit genügend Ressourcen (Fisch, Fleisch oder Pflanzen). Der…

### 12. [splitting_wins] Query `train-07217`
- **Question:** Wer gilt als die erste Erdgöttin?
- **Gold Answer(s):** Gaia | **Gold Article:** *Gott* (`7af2962db746a0bd`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=None, `parts_stem`=1, `dense`=1, `hyb_plain`=4, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** `erdgöttin` -> `erd + göttin`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `erd`, `gottin`
- **Gold Passage Snippet:** > ==== Erd- und Wassergötter ==== Darstellung der Gaia in einem syrischen Fresko um 730 Eine der ersten Theophanien der Erde und des Bodens war die einer Mutter, die mit Fruchtbarkeit in Verbindung gebracht wurde. Obwohl viele Erd- und einige Fruchtbarkeitsgötte…

### 13. [splitting_wins] Query `train-07594`
- **Question:** Wo werden in den USA Exekutivverordnungen der Behörden niedegreschrieben?
- **Gold Answer(s):** im Code of Federal Regulations | **Gold Article:** *Recht_der_Vereinigten_Staaten* (`f6389579745a68ed`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=73, `stem`=None, `parts_stem`=1, `dense`=2, `hyb_plain`=2, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** `exekutivverordnungen` -> `exekutiv + verordnungen`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `exekutiv`, `verordn`
- **Gold Passage Snippet:** > === Bundesrecht === Bundesrecht leitet sich von der Verfassung der Vereinigten Staaten ab, die alle gesetzgeberischen Kompetenzen auf der Bundesebene beim Kongress ansiedelt. Dabei sind die Kompetenzen, die überhaupt zur Bundesebene gehören, aufgrund des anges…

### 14. [splitting_wins] Query `train-07658`
- **Question:** Was waren die Gründe für die Nahrungsknappheit in Eritrea?
- **Gold Answer(s):** Dürre und wirtschaftspolitische Inkompetenz der autoritären Regierung | **Gold Article:** *Eritrea* (`aeff76c9b2ab69a3`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=48, `stem`=88, `parts_stem`=8, `dense`=2, `hyb_plain`=11, `hyb_parts`=2, `hyb_rewrite`=2
- **Query Compound Splits:** `nahrungsknappheit` -> `nahrung + knappheit`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `nahrung`
- **Gold Passage Snippet:** > === Landwirtschaft === Etwa 75 % der Bevölkerung sind in der Landwirtschaft beschäftigt. Trotzdem müssen Nahrungsmittel importiert werden, auch weil während des Krieges und darüber hinaus mindestens 300.000 Personen zum Militärdienst eingezogen waren und daher…

### 15. [splitting_wins] Query `train-09242`
- **Question:** Welches sind die wichtigsten Wirtschaftszweige in Boston?
- **Gold Answer(s):** Bildungseinrichtungen, Gesundheitswesen, Finanzwirtschaft und Technologie | **Gold Article:** *Boston
Boston  ist die größte Stadt in Neuengland und Hauptstadt des Bundesstaates Massachusetts an der Ostküste der Vereinigten Staaten. Die Metropole ist eine der ältesten, wohlhabendsten und kulturell reichsten Städte der USA. Sie beheimatet in ihrem Großraum sowohl die weltberühmte Harvard University als auch das ähnlich bedeutende MIT, beide in Cambridge gelegen. Im kulturellen Bereich sind die Symphony Hall und das in ihr residierende Boston Symphony Orchestra weltbekannt. Die Wirtschaftskraft der Region wird hauptsächlich durch Bildungseinrichtungen, Gesundheitswesen, Finanzwirtschaft und Technologie bestimmt.* (`69db5ad6ecf67430`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=19, `stem`=38, `parts_stem`=9, `dense`=2, `hyb_plain`=3, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** `wirtschaftszweige` -> `wirtschaft + zweige`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `wirtschaft`
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Wirtschaftszweige Boston Industrie Dienstleistungen Technologie Handel`
- **Gold Passage Snippet:** > Boston  ist die größte Stadt in Neuengland und Hauptstadt des Bundesstaates Massachusetts an der Ostküste der Vereinigten Staaten. Die Metropole ist eine der ältesten, wohlhabendsten und kulturell reichsten Städte der USA. Sie beheimatet in ihrem Großraum sowo…

### 16. [splitting_losses] Query `train-00019`
- **Question:** Was ist das einzige Beweismaterial für das Wort Israel?
- **Gold Answer(s):** die ägyptische Merenptah-Stele | **Gold Article:** *Israel* (`336740857a3716ab`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=6, `stem`=6, `parts_stem`=14, `dense`=4, `hyb_plain`=1, `hyb_parts`=2, `hyb_rewrite`=2
- **Query Compound Splits:** `beweismaterial` -> `beweis + material`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Israel Beweis Israel Geschichte Dokument`
- **Gold Passage Snippet:** > == Name == Der älteste Beleg für das Wort „Israel“ ist die ägyptische Merenptah-Stele, die sich heute im Ägyptischen Museum in Kairo befindet. Sie beschreibt einen Feldzug gegen ein Volk Israel im Lande Kanaan und wird auf das Jahr 1211 v. Chr. datiert. Die Vo…

### 17. [splitting_losses] Query `train-00272`
- **Question:** Seit wann wird der Orden of the British Empire verliehen?
- **Gold Answer(s):** 4. Juni 1917 | **Gold Article:** *Order_of_the_British_Empire
, kurz , ist ein britischer Ritterorden, der am 4. Juni 1917 von König Georg V. gestiftet wurde. Die Aufnahme in den Orden würdigt Verdienste in den Bereichen Kunst und Wissenschaft, der Arbeit für gemeinnützige und Wohlfahrtsorganisationen sowie den öffentlichen Dienst. Der Orden ist in eine militärische und eine zivile Abteilung unterteilt und umfasst fünf Ränge, deren obere beide ihre Mitglieder unter besonderen Voraussetzungen in den Ritterstand erheben. Die Anzahl der Mitglieder ist zahlenmäßig nicht begrenzt. Der Orden ist der jüngste und mitgliederstärkste britische Ritterorden und hat die meisten ausländischen Mitglieder.* (`60224de76e20fe62`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=7, `stem`=10, `parts_stem`=11, `dense`=1, `hyb_plain`=3, `hyb_parts`=5, `hyb_rewrite`=5
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gold Passage Snippet:** > , kurz , ist ein britischer Ritterorden, der am 4. Juni 1917 von König Georg V. gestiftet wurde. Die Aufnahme in den Orden würdigt Verdienste in den Bereichen Kunst und Wissenschaft, der Arbeit für gemeinnützige und Wohlfahrtsorganisationen sowie den öffentlic…

### 18. [splitting_losses] Query `train-00333`
- **Question:** Welche israelischen Städte verfügen über große Häfen?
- **Gold Answer(s):** Haifa, Ashdod und Eilat | **Gold Article:** *Israel* (`6b63ea56bb1e25db`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=5, `stem`=9, `parts_stem`=21, `dense`=8, `hyb_plain`=3, `hyb_parts`=7, `hyb_rewrite`=7
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gold Passage Snippet:** > === Flüsse und Meere === Israel grenzt an zwei Meere: an das Mittelmeer im Westen und an das Rote Meer im Süden. In Haifa, Ashdod und Eilat gibt es teils große Häfen, die ein wichtiger Teil der israelischen Wirtschaft sind. Die wichtigsten Wasserquellen Israel…

### 19. [splitting_losses] Query `train-00463`
- **Question:** Welches Unternehmen betreibt die Straßenbahn in Straßburg?
- **Gold Answer(s):** Compagnie des transports strasbourgeois (CTS) | **Gold Article:** *Straßburg* (`bbf10b4355af957e`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=5, `stem`=3, `parts_stem`=12, `dense`=2, `hyb_plain`=3, `hyb_parts`=5, `hyb_rewrite`=5
- **Query Compound Splits:** `straßenbahn` -> `straße + bahn`, `straßburg` -> `straße + burg`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `bahn`, `burg`, `strass`
- **Gold Passage Snippet:** > === ÖPNV === Die Straßenbahn Straßburg, welche von der Compagnie des transports strasbourgeois (CTS) betrieben wird, besteht aus sechs Linien (A bis F), die neben den zahlreichen Buslinien den Kern des öffentlichen Nahverkehrs ausmachen. Eine Besonderheit dies…

### 20. [splitting_losses] Query `train-02209`
- **Question:** Seit wann ist es möglich mit der Straßenbahn von Straßburg nach Deutschland zu fahren?
- **Gold Answer(s):** 29. April 2017  | **Gold Article:** *Straßburg* (`bbf10b4355af957e`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=1, `stem`=4, `parts_stem`=21, `dense`=1, `hyb_plain`=1, `hyb_parts`=7, `hyb_rewrite`=7
- **Query Compound Splits:** `straßenbahn` -> `straße + bahn`, `straßburg` -> `straße + burg`, `deutschland` -> `deutsch + land`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `bahn`, `burg`, `deutsch`, `strass`
- **Gold Passage Snippet:** > === ÖPNV === Die Straßenbahn Straßburg, welche von der Compagnie des transports strasbourgeois (CTS) betrieben wird, besteht aus sechs Linien (A bis F), die neben den zahlreichen Buslinien den Kern des öffentlichen Nahverkehrs ausmachen. Eine Besonderheit dies…

### 21. [splitting_losses] Query `train-02923`
- **Question:** Wann endete die Aberdeen Regierung? 
- **Gold Answer(s):** Im Februar 1855 | **Gold Article:** *Krimkrieg* (`03efff8b013cb82a`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=16, `stem`=10, `parts_stem`=12, `dense`=38, `hyb_plain`=6, `hyb_parts`=6, `hyb_rewrite`=6
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gold Passage Snippet:** > === Großbritannien === Der Krimkrieg zeigte, dass es erhebliche Missstände im britischen Militär gab. Dadurch verlor die Regierung Aberdeen erheblich an Ansehen. Im Februar 1855 wurde sie zum Rücktritt gezwungen, und Palmerston übernahm die Bildung eines neuen…

### 22. [splitting_losses] Query `train-03234`
- **Question:** Beim Abbau von was haben Insekten in Ökosystem großen Anteil?
- **Gold Answer(s):** organischer Stoffe | **Gold Article:** *Insekten* (`7bf10263c888f78b`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=27, `stem`=7, `parts_stem`=20, `dense`=2, `hyb_plain`=4, `hyb_parts`=4, `hyb_rewrite`=4
- **Query Compound Splits:** `ökosystem` -> `öko + system`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Insekten Abbau Ökosystem Anteil`
- **Gold Passage Snippet:** > == Lebensweise == Aufgrund ihrer Vielfalt haben Insekten heute beinah jede ihrer Größe angemessene ökologische Nische realisiert. Dabei spielt eine große Anzahl der Arten eine bedeutende Rolle bei der Remineralisierung organischer Stoffe im Boden, in der Boden…

### 23. [splitting_losses] Query `train-04233`
- **Question:** Welcher Fußballclub ist unter den drei Vereinen mit den meisten eingetragenen Mitgliedern der Welt?
- **Gold Answer(s):** Benfica Lissabon | **Gold Article:** *FC_Barcelona* (`4196d9a47d9e9b2f`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=4, `stem`=2, `parts_stem`=36, `dense`=7, `hyb_plain`=1, `hyb_parts`=8, `hyb_rewrite`=8
- **Query Compound Splits:** `fußballclub` -> `fußball + club`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Fußballclub Mitglieder Welt Vereine`
- **Gold Passage Snippet:** > === „Socis“ === Die Mitgliederzahl des FC Barcelona seit Bestehen des Vereins Wesentlichen Einfluss auf die Vereinspolitik haben die ''socis'' (span.: ''socios''), die Mitglieder des FC Barcelona. So wird der Präsident des FC Barcelona durch die Stimmen der Mi…

### 24. [splitting_losses] Query `train-06827`
- **Question:** Was ist eine Flechte?
- **Gold Answer(s):** eine symbiotische Lebensgemeinschaft zwischen einem Pilz und Grünalgen oder Cyanobakterien | **Gold Article:** *Symbiose* (`d1fa200a51b1089c`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=7, `parts_stem`=11, `dense`=9, `hyb_plain`=21, `hyb_parts`=2, `hyb_rewrite`=2
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gold Passage Snippet:** > == Häufigkeit von Symbiosen == Der größte Teil der Biomasse auf der Erde besteht aus symbiotischen Systemen, da ein großer Teil der Bäume und Sträucher auf Bestäubung durch andere Spezies angewiesen ist. Hinzu kommen die Flechten, eine symbiotische Lebensgemei…

### 25. [splitting_losses] Query `train-07041`
- **Question:** In welcher Distanz können M270 Raketen Ziele erreichen?
- **Gold Answer(s):** bis zu 70 Kilometern | **Gold Article:** *United_States_Army* (`53eea733f579a3e7`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=10, `stem`=4, `parts_stem`=13, `dense`=7, `hyb_plain`=5, `hyb_parts`=6, `hyb_rewrite`=6
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gold Passage Snippet:** > ==== Artillerie ==== Knapp 1.000 Mehrfach-Raketen-Artilleriesysteme des Typs M270 dienen der punktgenauen Bekämpfung von gepanzerten Zielen. Je nach Version haben sie eine Reichweite von bis zu 70 Kilometern Entfernung und können mit entsprechender Streumuniti…

### 26. [hybrid_preprocessing_wins_h3] Query `train-01238`
- **Question:** Welcher Schweizer Bahnhof wird am meisten genutzt? 
- **Gold Answer(s):** Zürcher Hauptbahnhof | **Gold Article:** *Bern* (`cd9619bafd6e4ac4`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=None, `parts_stem`=72, `dense`=5, `hyb_plain`=21, `hyb_parts`=9, `hyb_rewrite`=9
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gold Passage Snippet:** > ==== Öffentlicher Verkehr ==== Mit Erdgas betriebener Bus in der Altstadt Der Berner Hauptbahnhof ist ein bedeutender Eisenbahn- und Busknotenpunkt. Er erzielt nach dem Zürcher Hauptbahnhof mit 150'000 Bahnreisenden pro Tag die schweizweit höchste Benutzerfreq…

### 27. [hybrid_preprocessing_wins_h3] Query `train-01707`
- **Question:** Wie heißt das bekannteste Filmfestival in Ägypten? 
- **Gold Answer(s):** 'Cairo International Film Festival | **Gold Article:** *Ägypten* (`1a79c27e95465959`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=9, `parts_stem`=1, `dense`=2, `hyb_plain`=22, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** `filmfestival` -> `film + festival`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `festival`, `film`
- **Gold Passage Snippet:** > === Kunst und Architektur === Ägypten war eine der ersten Zivilisationen, der es gelang Gestaltungselemente in Kunst und Architektur zu chiffrieren. Die im Dienste der Pharaonen geschaffenen Wandmalereien folgten einem rigiden Code visueller Regeln und Bedeutu…

### 28. [hybrid_preprocessing_wins_h3] Query `train-02069`
- **Question:** Was versteht man unter "Browserkrieg"?
- **Gold Answer(s):** Mit Erscheinen des ''Internet Explorers'' begann ein Verdrängungswettbewerb zwischen den Browser-Herstellern Microsoft und Netscape | **Gold Article:** *Webbrowser* (`c8caad913627c76e`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=None, `parts_stem`=20, `dense`=7, `hyb_plain`=14, `hyb_parts`=3, `hyb_rewrite`=3
- **Query Compound Splits:** `browserkrieg` -> `browser + krieg`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `brows`, `krieg`
- **Gold Passage Snippet:** > === Internet Explorer === Aufgrund des Erfolgs des ''Netscape Navigators'' brachte Microsoft, das bis dahin das Internet unterschätzt hatte, 1995 seinen ''Internet Explorer'' heraus, der nicht selbst entwickelt, sondern vom Unternehmen Spyglass (NCSA Mosaic) e…

### 29. [hybrid_preprocessing_wins_h3] Query `train-02349`
- **Question:** Welcher Stadt des Britischen Weltreiches wollte sich in den Zweiten Weltkrieg nicht einmischen?
- **Gold Answer(s):** der Irische Freistaat | **Gold Article:** *Britisches_Weltreich* (`861803146ff4fc4e`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=10, `parts_stem`=13, `dense`=2, `hyb_plain`=13, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** `weltreiches` -> `welt + reiches`, `weltkrieg` -> `welt + krieg`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `krieg`, `reich`, `welt`
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Britische Kolonien Zweiter Weltkrieg Nicht einmischen`
- **Gold Passage Snippet:** > === Zweiter Weltkrieg === Großbritanniens Kriegserklärung an das von den Nationalsozialisten beherrschte Deutsche Reich im September 1939 schloss die Kronkolonien und Indien mit ein, jedoch nicht automatisch die Dominions. Kanada, Südafrika, Australien und Neu…

### 30. [hybrid_preprocessing_wins_h3] Query `train-03611`
- **Question:** Was lösen die Steine in Gletschern aus?
- **Gold Answer(s):** in der Regel einen Schliff vergleichbar mit der Wirkung von Schleifpapier, während größere Partikel deutliche Kratzspuren und Rillen im Fels hinterlassen können | **Gold Article:** *Gletscher* (`a670703d67032b4e`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=75, `parts_stem`=42, `dense`=2, `hyb_plain`=20, `hyb_parts`=10, `hyb_rewrite`=4
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Gletscherspaltenbildung Eisbruch Eisbewegung Gletscherausbruch Eisbruchmechanismen`
- **Gold Passage Snippet:** > ==== Gletscherschliff und Gletscherschrammen ==== Im Gletschereis mitgeführtes Gesteinsmaterial verschiedener Korngrößen – von feinem Ton bis zu mehrere Meter messenden Findlingen – kann im Gesteinsuntergrund deutliche Spuren hinterlassen. Feinkörniges Materia…

### 31. [hybrid_preprocessing_wins_h3] Query `train-05190`
- **Question:** Wie viel Prozent des Stromverbrauchs auf St.Helen wird durch Windkraft gedeckt?
- **Gold Answer(s):** 20 bis 30 % | **Gold Article:** *St._Helena__Insel_* (`60c7167610907f54`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=None, `parts_stem`=2, `dense`=1, `hyb_plain`=15, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** `stromverbrauchs` -> `strom + verbrauchs`, `windkraft` -> `wind + kraft`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `kraft`, `strom`, `wind`
- **Gold Passage Snippet:** > === Energieversorgung === Im Jahr 2017 wurden rund 75 % des Stroms auf St. Helena durch sechs Dieselgeneratoren des Kraftwerks in Rupert’s mit einer Leistung von 7600 kW erzeugt. Daneben werden 20 bis 30 % der Elektrizität durch zwölf Windkrafträder in der Dea…

### 32. [hybrid_preprocessing_wins_h3] Query `train-06969`
- **Question:** Wie schnell darf man auf der Schweizer Autobahn fahren?
- **Gold Answer(s):** 60 und 100 km/h | **Gold Article:** *Bern* (`c564ad7c94fa3127`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=81, `parts_stem`=49, `dense`=2, `hyb_plain`=16, `hyb_parts`=6, `hyb_rewrite`=6
- **Query Compound Splits:** `autobahn` -> `auto + bahn`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `auto`, `bahn`
- **Gold Passage Snippet:** > ==== Strassenverkehr ==== Die Stadt Bern besitzt seit den 1970er-Jahren eine Autobahnumfahrung, die den innerstädtischen Autoverkehr entlastet und Anschlüsse an alle wichtigen Schweizer Autobahnen hat. Die A1 verbindet Bern mit Genf und Zürich, die A12 mit Fre…

### 33. [hybrid_preprocessing_wins_h3] Query `train-08156`
- **Question:** Wie nennt man die portugiesische Kultur des Tanzes? 
- **Gold Answer(s):** Nova Dança Portuguesa | **Gold Article:** *Portugal* (`d65ca2e83fc34509`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=60, `parts_stem`=7, `dense`=1, `hyb_plain`=15, `hyb_parts`=1, `hyb_rewrite`=1
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `kultur`
- **Gold Passage Snippet:** > === Tanz === Im Bereich des Kunsttanzes ist Portugal seit Anfang der 1990er-Jahre recht erfolgreich und gehört mit Russland zu den wichtigsten Ländern in Europa auf diesem Gebiet. Zahlreiche Tänzer sowie Choreographen konnten europaweiten oder weltweiten Ruhm …

### 34. [hybrid_preprocessing_wins_h3] Query `train-08266`
- **Question:** Was ist die Amtssprache von Eritrea?
- **Gold Answer(s):** Es gibt keine offiziell festgelegte Amtssprache | **Gold Article:** *Eritrea* (`3a4cf9d876456270`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=49, `stem`=54, `parts_stem`=12, `dense`=2, `hyb_plain`=13, `hyb_parts`=6, `hyb_rewrite`=6
- **Query Compound Splits:** `amtssprache` -> `amt + sprache`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `amt`, `sprach`
- **Gold Passage Snippet:** > === Sprachen === Die neun Sprachen der neun größten Ethnien gelten formell als gleichberechtigte Nationalsprachen. Diese sind  Tigrinya (ትግርኛ) (2,3 Millionen Sprecher), Tigre (800.000), Afar (300.000), Saho, Kunama, Bedscha, Blin, Nara (je rund 100.000) und Ar…

### 35. [hybrid_preprocessing_wins_h3] Query `train-08689`
- **Question:** Was nennt man im Surfen im Internet?
- **Gold Answer(s):** das aufeinanderfolgende Abrufen beliebiger Hyperlinks als Verbindung zwischen Webseiten | **Gold Article:** *Webbrowser
Webbrowser oder allgemein auch Browser (engl. , '','' ‚stöbern, schmökern, umsehen‘, auch ‚abgrasen‘) sind spezielle Computerprogramme zur Darstellung von Webseiten im ''World Wide Web'' oder allgemein von Dokumenten und Daten. Das Durchstöbern des ''World Wide Webs'' beziehungsweise das aufeinanderfolgende Abrufen beliebiger Hyperlinks als Verbindung zwischen Webseiten mit Hilfe solch eines Programms wird auch als Internetsurfen bezeichnet. Neben HTML-Seiten können Webbrowser verschiedene andere Arten von Dokumenten wie zum Beispiel Bilder und PDF-Dokumente anzeigen. Webbrowser stellen die Benutzeroberfläche für Webanwendungen dar.* (`fbb1c825aa40ede8`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=None, `parts_stem`=7, `dense`=2, `hyb_plain`=12, `hyb_parts`=3, `hyb_rewrite`=3
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `internet`, `surf`
- **Gold Passage Snippet:** > Webbrowser oder allgemein auch Browser (engl. , '','' ‚stöbern, schmökern, umsehen‘, auch ‚abgrasen‘) sind spezielle Computerprogramme zur Darstellung von Webseiten im ''World Wide Web'' oder allgemein von Dokumenten und Daten. Das Durchstöbern des ''World Wid…

### 36. [llm_rewrite_wins_h5] Query `train-00929`
- **Question:** Wer leitete das erste Verfassungskommitee im Osmanischen Reich?
- **Gold Answer(s):** Midhat Pascha | **Gold Article:** *Osmanisches_Reich* (`020758c550971c73`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=99, `parts_stem`=71, `dense`=1, `hyb_plain`=28, `hyb_parts`=12, `hyb_rewrite`=4
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Erster Verfassungskomitee Osmanisches Reich`
- **Gold Passage Snippet:** > ==== Erste Verfassungsperiode: 1876 bis 1878 ==== Titelblatt der Osmanischen Verfassung (''Kanûn-ı Esâsî'') von 1876 Die Osmanische Verfassung von 1876 ist neben den Tanzimat-Edikten der dritte Schritt in den großen Reformen des 19. Jahrhunderts. Sie wird als …

### 37. [llm_rewrite_wins_h5] Query `train-01081`
- **Question:** Woher kommt der Name Mexikos? 
- **Gold Answer(s):** von der jetzigen Hauptstadt | **Gold Article:** *Mexiko-Stadt* (`450abc1cd32ce203`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=None, `parts_stem`=None, `dense`=1, `hyb_plain`=13, `hyb_parts`=13, `hyb_rewrite`=3
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `mexikanische Herkunft indigene Namen sprachliche Ursprünge kulturelle Namensgebung`
- **Gold Passage Snippet:** > == Der Name der Stadt == Die Mexikaner nennen ihre Hauptstadt meist ''el D.F.'' („el De-Efe“, Abkürzung von ''Distrito Federal'', „Bundesbezirk“). Wenn also von ''México'' die Rede ist, ist normalerweise der Staat gemeint. Weniger häufig und meist in offiziell…

### 38. [llm_rewrite_wins_h5] Query `train-03119`
- **Question:** Welche künstlerischen Werke wurden durch die Auseinandersetzung in der Aufklärung von den Wissenschaftler mehr beachtet?
- **Gold Answer(s):** Romane, Dramen und Gedichte, Bilder und Kompositionen | **Gold Article:** *Aufklärung* (`4b59a51cce532bc7`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=None, `parts_stem`=None, `dense`=1, `hyb_plain`=14, `hyb_parts`=13, `hyb_rewrite`=8
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Aufklärung Kunst Werke Wissenschaftler Beachtung`
- **Gold Passage Snippet:** > == Kunst == Gegenüber den Wissenschaften (französisch den „lettres“, lateinisch der „Literatur“) und den Künsten (den „artes“ als dem Bereich technischer Erfindungen) emanzipierten sich im Lauf des 17. Jahrhunderts die zwei komplementären Märkte und Bildungsge…

### 39. [llm_rewrite_wins_h5] Query `train-03241`
- **Question:** Wer ist Gabe Newell?
- **Gold Answer(s):** Geschäftsführer der Valve Corporation | **Gold Article:** *Microsoft_Windows_8* (`30fd5989e1ef4a27`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=1, `stem`=1, `parts_stem`=12, `dense`=None, `hyb_plain`=1, `hyb_parts`=27, `hyb_rewrite`=4
- **Query Compound Splits:** `newell` -> `new + ell`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `ell`, `new`
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Gabe Newell Gabe Newell Computer Hardware`
- **Gold Passage Snippet:** > === Windows Store === Modern-Apps können über den neuen ''Windows Store'' bezogen werden. Microsoft sieht In-App-Käufe (das Erweitern von installierten Programmen um weitere Funktionen direkt aus diesen heraus) vor und bietet eine Möglichkeit, die Anwendungen …

### 40. [llm_rewrite_wins_h5] Query `train-03371`
- **Question:** Welche Territorien gehören zum Griechenland?
- **Gold Answer(s):** griechischen Festland am südlichen Ende des Balkans, der Halbinsel Peloponnes, die jedoch durch den Bau des Kanals von Korinth (eingeweiht 1893) vom Festland getrennt wurde, sowie zahlreichen Inseln zusammen, die hauptsächlich in der Ägäis, im Ionischen, aber auch im Libyschen Meer liegen | **Gold Article:** *Griechenland* (`2075bbe735e478b0`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=86, `parts_stem`=74, `dense`=2, `hyb_plain`=27, `hyb_parts`=12, `hyb_rewrite`=3
- **Query Compound Splits:** `griechenland` -> `griechen + land`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `griech`, `land`
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Griechenland Territorien Inseln Archipel Region`
- **Gold Passage Snippet:** > == Geographie == Griechenland liegt am östlichen Mittelmeer in Südeuropa und setzt sich geographisch aus dem griechischen Festland am südlichen Ende des Balkans, der Halbinsel Peloponnes, die jedoch durch den Bau des Kanals von Korinth (eingeweiht 1893) vom Fe…

### 41. [llm_rewrite_wins_h5] Query `train-03565`
- **Question:** Was versteht man unter NES?
- **Gold Answer(s):** eine stationäre 8-Bit-Spielkonsole des japanischen Spielkonsolenherstellers Nintendo | **Gold Article:** *Nintendo_Entertainment_System
Das Nintendo Entertainment System (kurz meist NES genannt) ist eine stationäre 8-Bit-Spielkonsole des japanischen Spielkonsolenherstellers Nintendo. Die Konsole erschien in Nordamerika, Brasilien, Europa, Asien und Australien. Die japanische Version wurde als Famicom (kurz für ''Family Computer'') vermarktet. Das NES war mit über 60 Millionen verkauften Einheiten lange Zeit die meistverkaufte Konsole überhaupt. Nach dem Zusammenbruch der Videospielebranche im Jahre 1983, dem sogenannten „Video Game Crash“, belebte Nintendo den Heimkonsolenmarkt mit ihrer Konsole neu. Der Controller führte das Steuerkreuz (kurz: ''D-Pad'' für ''Direction-Pad'') ein und definierte die grundsätzliche Tastenanordnung der Gamepads. Bekannte Spieleserien wie Super Mario Bros., The Legend of Zelda, Metroid oder Castlevania gaben auf dem NES ihr Heimkonsolen-Debüt.* (`1b51223ac910434c`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=70, `stem`=66, `parts_stem`=None, `dense`=1, `hyb_plain`=11, `hyb_parts`=12, `hyb_rewrite`=7
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Netzwerk-Energie-System NES`
- **Gold Passage Snippet:** > Das Nintendo Entertainment System (kurz meist NES genannt) ist eine stationäre 8-Bit-Spielkonsole des japanischen Spielkonsolenherstellers Nintendo. Die Konsole erschien in Nordamerika, Brasilien, Europa, Asien und Australien. Die japanische Version wurde als …

### 42. [llm_rewrite_wins_h5] Query `train-04907`
- **Question:** Welcher Fluss fließt durch Philadelphia?
- **Gold Answer(s):** Delaware River | **Gold Article:** *Philadelphia
Philadelphia ist eine Stadt im US-Bundesstaat Pennsylvania. Mit rund 1,6 Millionen Einwohnern (Stand: 2016, Schätzung des U.S. Census Bureau) ist sie die sechstgrößte Stadt der Vereinigten Staaten und die größte des Bundesstaates Pennsylvania. An der Ostküste ist Philadelphia nach New York City die zweitgrößte Stadt. Die Stadt liegt am Delaware River im Zentrum der Metropolregion Delaware Valley.
In der Geschichte der USA ist Philadelphia eine der bedeutendsten Städte. Nach New York und vor Washington war sie 1790 bis 1800 Nationalhauptstadt und damals die größte Stadt der USA sowie nach London die zweitgrößte englischsprachige Stadt der Welt. In Philadelphia tagte der erste und teilweise auch der zweite Kontinentalkongress sowie der Verfassungskonvent von 1787, die Amerikanische Unabhängigkeitserklärung (4. Juli 1776) wurde hier verkündet und die Verfassung beschlossen.
Philadelphia wird umgangssprachlich ''Philly'' oder ''City of Brotherly Love'' genannt. Der Name der Stadt ist dem mehrerer antiker Städte (, von ''Philádelphos'', dem Beinamen Attalos II.) nachempfunden und wurde von William Penn wohl auch direkt auf das unabhängig gebildete Substantiv () bezogen. Er ist zusammengesetzt aus den Wurzeln von  oder  und letztlich  sowie , bedeutet also eben ‚Ort brüderlicher Liebe‘ oder ‚Bruderliebe‘.* (`53a4279b91d963a5`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=85, `stem`=85, `parts_stem`=90, `dense`=1, `hyb_plain`=12, `hyb_parts`=13, `hyb_rewrite`=3
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Fluss Philadelphia Delaware River`
- **Gold Passage Snippet:** > Philadelphia ist eine Stadt im US-Bundesstaat Pennsylvania. Mit rund 1,6 Millionen Einwohnern (Stand: 2016, Schätzung des U.S. Census Bureau) ist sie die sechstgrößte Stadt der Vereinigten Staaten und die größte des Bundesstaates Pennsylvania. An der Ostküste …

### 43. [llm_rewrite_wins_h5] Query `train-05648`
- **Question:** Was sind die größten Sport Clubs Uruguays?
- **Gold Answer(s):**  Nacional und Peñaro | **Gold Article:** *Montevideo* (`c2b95354701e91c1`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=None, `parts_stem`=None, `dense`=1, `hyb_plain`=15, `hyb_parts`=19, `hyb_rewrite`=5
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `größte Sportclubs Uruguay Fußballvereine Uruguay Sport`
- **Gold Passage Snippet:** > == Sport == Montevideo ist auch in sportlicher Hinsicht die zentrale Stadt des Landes. So stammt etwa der überwiegende Teil der im Profifußball vertretenen Vereine aus der Hauptstadt. So haben neben den beiden dominierenden, großen Klubs des Landes, Nacional u…

### 44. [llm_rewrite_wins_h5] Query `train-06148`
- **Question:** Welches Gebiet wurde von der Lhasas-Regierung regiert?
- **Gold Answer(s):** „politischen Tibet“ | **Gold Article:** *Tibet* (`1390586626344ca5`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=18, `stem`=46, `parts_stem`=None, `dense`=2, `hyb_plain`=1, `hyb_parts`=14, `hyb_rewrite`=2
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Lhasa Regierung Gebiet Verwaltung`
- **Gold Passage Snippet:** > === Autonomes Gebiet Tibet === Detaillierte Karte des Tibetischen Kulturraums und der autonomen tibetischen Verwaltungsgliederungen der VR China Das ''Autonome Gebiet Tibet'' ist eine Verwaltungseinheit der Volksrepublik China. Es umfasst ein Gebiet von 1,2 Mi…

### 45. [llm_rewrite_wins_h5] Query `train-06334`
- **Question:** Wie groß ist North Carolina?
- **Gold Answer(s):** 139.389 Quadratkilometern (Landfläche 125.919 km²) | **Gold Article:** *North_Carolina* (`a8d074854ec0ff03`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=47, `parts_stem`=44, `dense`=1, `hyb_plain`=44, `hyb_parts`=11, `hyb_rewrite`=3
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Größe North Carolina Fläche Einwohner`
- **Gold Passage Snippet:** > === Lage und Ausdehnung === North Carolina liegt an der Südostküste der Vereinigten Staaten und grenzt an die Bundesstaaten South Carolina und Georgia im Süden, Tennessee im Westen und Virginia im Norden. Im Osten liegt der Atlantik. Die Nordgrenze des Staats …

### 46. [residual_hard_failures] Query `train-01641`
- **Question:** Warum gehen gut gelaunte Menschen höhere Risiken ein? 
- **Gold Answer(s):** da das Risiko eines negativen Ausgangs der Situation gerne unterschätzt wird | **Gold Article:** *Emotion* (`d61a32354de391b0`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=None, `parts_stem`=None, `dense`=11, `hyb_plain`=27, `hyb_parts`=30, `hyb_rewrite`=30
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gold Passage Snippet:** > === Urteile und Entscheidungen === Emotionen beeinflussen die Beurteilung, ob etwas positiv oder negativ, nützlich oder bedrohlich ist. Beurteilungen fallen dabei positiver aus, wenn die Stimmung positiv ist. Befindet man sich in einer positiven Stimmung, werd…

### 47. [residual_hard_failures] Query `train-03433`
- **Question:** Wann hat man angefangen Papier in Europa zu benutzen?
- **Gold Answer(s):** im 12. Jahrhundert | **Gold Article:** *Chinesische_Schrift* (`00f5bd6c9cab2e08`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=36, `stem`=20, `parts_stem`=52, `dense`=19, `hyb_plain`=11, `hyb_parts`=15, `hyb_rewrite`=15
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gold Passage Snippet:** > === Vom Altertum zur Neuzeit === Als besondere Schriftsubstrate des chinesischen Altertums gelten Seide und Bambus, der in schmale Brettchen geschnitten beschrieben wurde (wie Guodian-Bambustexte). Beide findet man im europäischen Altertum nicht, wo die Verwen…

### 48. [residual_hard_failures] Query `train-04970`
- **Question:** Woher kommt Hafis? 
- **Gold Answer(s):** Persien | **Gold Article:** *Iran* (`306e014ecb964611`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=33, `stem`=33, `parts_stem`=33, `dense`=None, `hyb_plain`=67, `hyb_parts`=67, `hyb_rewrite`=22
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `Hafis Herkunft Hafis Ursprung Hafis Name`
- **Gold Passage Snippet:** > === Literatur === Persien, hierbei insbesondere das südliche Fars, weist in der Dichtkunst zahlreiche Berühmtheiten auf, von denen Firdausi, Hafis und Saadi einige der bekanntesten sind. In der Neuzeit gewann die Prosa in der persischen Literatur zunehmende Be…

### 49. [residual_hard_failures] Query `train-08730`
- **Question:** Wer setzte Napoleons Eroberungen 1815 ein Ende?
- **Gold Answer(s):** eine Koalition europäischer Armeen | **Gold Article:** *Britisches_Weltreich* (`267ae42c6b1a9450`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=None, `stem`=30, `parts_stem`=27, `dense`=14, `hyb_plain`=29, `hyb_parts`=13, `hyb_rewrite`=13
- **Query Compound Splits:** *(none in query; compound split occurred in passage)*
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** *(none)*
- **Gold Passage Snippet:** > === Napoleonische Kriege und Abschaffung der Sklaverei === Während der Herrschaft Napoleon Bonapartes wurde Großbritannien im sogenannten Britisch-Französischen Kolonialkonflikt erneut durch das neu entstandene französische Kaiserreich herausgefordert. Im Gege…

### 50. [residual_hard_failures] Query `train-08788`
- **Question:** In welcher Sprache wurde an der Uni Straßburg im 18. Jahrhundert unterrichtet?
- **Gold Answer(s):** auf Deutsch  | **Gold Article:** *Elsass* (`efe1b403d2bed284`)
- **Gold Ranks (`own_hard_negatives_dropped`):** `plain`=62, `stem`=95, `parts_stem`=None, `dense`=14, `hyb_plain`=9, `hyb_parts`=35, `hyb_rewrite`=35
- **Query Compound Splits:** `straßburg` -> `straße + burg`
- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** `burg`, `strass`
- **Gold Passage Snippet:** > === Französisches Königreich 1648–1789 === Zwischen 1633 und 1681 übernahm das Königreich Frankreich nach und nach, teils durch Verträge (de jure), teils durch Annexion (de facto), in den meisten elsässischen Regionen die Landesherrschaft, meist aber nicht die…
