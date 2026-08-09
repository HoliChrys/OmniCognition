# Événements comme schémas dans le RAG — rapport de recherche

> Comment intégrer la notion d'**event** dans un RAG/mémoire : un event est un
> **hub** avec un **intervalle temporel** (début + fin *hypothétique/ouverte*),
> autour duquel des **faits gravitent**, et dont le **type** (`event:type:*`)
> définit une **typologie récurrente** de rôles/slots (ex. *war* ⇒
> belligérants, territoires/lands, localisations → camps/fronts, casus belli,
> timeline, pertes, traités) — typologie qui engendre les **sous-questions**
> auxquelles répondre par un retrieval scopé.

Synthèse de 5 axes, sources arXiv/ACL/peer-reviewed prioritaires. À la fin :
un **blueprint concret pour MetaCog-Mem** reliant nos briques existantes
(`apply_pull`, tags `event:type:*`, tag navigator, date hard-scope, clue_search).

---

## 1. L'event comme objet à **extension temporelle** (début + fin hypothétique)

- **Algèbre d'intervalles d'Allen** : tout event = un intervalle ; 13 relations
  de base (precedes, meets, overlaps, starts, finishes, during + inverses +
  equals) → algèbre fermée pour raisonner sur les extensions temporelles.
  [TLEX (arXiv:2406.05265)](https://arxiv.org/html/2406.05265)
- **TimeML / TIMEX3** ancre l'extension via DATE/TIME/DURATION (+ SET pour le
  récurrent) et 14 TLINK ≈ relations d'Allen. **TLEX** matérialise un
  `[start, end]` par event en **scindant chaque intervalle en points
  start/end** (Point Algebra) puis tri topologique → timeline exacte. Son
  output **trunk-and-branch sépare le réel des branches modales/contrefactuelles
  /niées** — c'est exactement la distinction « fin hypothétique / event
  hypothétique ». [TLEX](https://arxiv.org/html/2406.05265)
- **Temporal KGs** : deux régimes — *event-style* (`t_start = t_end`, quadruple
  `(h,r,t,τ)` : ICEWS, GDELT) vs *interval-based* (`⟨s,r,o,[t_start,t_end]⟩` :
  YAGO11k, Wikidata12k). Les faits d'intervalle se classent par **ouverture** :
  *left-open* (start inconnu), **right-open (end inconnu = ongoing/encore vrai)**,
  ou *closed*. → **le right-open est la primitive pour modéliser une fin
  ouverte.** [Survey on TKG (arXiv:2403.04782)](https://arxiv.org/html/2403.04782v1)
- **Inférer** l'intervalle plutôt que le supposer donné : **TEMT** prédit
  `[start,end]` en fusionnant représentations PLM + embeddings temporels, en
  inductif sur entités inconnues.
  [TEMT (arXiv:2309.16357)](https://arxiv.org/abs/2309.16357)
- **Mémoire épisodique** : stocker des entrées ponctuelles cause une
  **« fragmentation temporelle »** (durées et états *ongoing* irrécupérables).
  **TSM** sépare *dialogue time* (quand c'est dit) de *semantic time* (quand
  l'event a lieu/tient), construit une *Durative Memory* (intervalles consolidés
  par GMM) ; le ranking met l'**overlap d'intervalle en clé primaire**, la
  similarité sémantique en tie-break → +22.56 % sur les questions temporelles.
  [TSM (arXiv:2601.07468)](https://arxiv.org/html/2601.07468v1) ·
  [Episodic Memory position (arXiv:2502.06975)](https://arxiv.org/abs/2502.06975)
- **Scoper le retrieval par le temps** : le RAG sémantique nu échoue car
  « revenue 2021 » vs « 2022 » ont des embeddings quasi identiques. **TG-RAG**
  extrait à la requête l'ensemble de timestamps `T^q` et met **score 0 à tout
  edge `τ ∉ T^q`** (filtre temporel dur sur PPR) + graphe temporel hiérarchique
  (année→…→jour). **STAR-RAG** pondère par **décroissance exponentielle** des
  spans inter-events.
  [TG-RAG (arXiv:2510.13590)](https://arxiv.org/html/2510.13590v1) ·
  [STAR-RAG (arXiv:2510.16715)](https://arxiv.org/html/2510.16715v1)

**À retenir** : matérialiser start/end par event ; **right-open = ongoing** ;
edges timestampés distincts (anti-collapse d'embedding) ; **scope temporel dur**
(τ∉T ⇒ 0) avec l'overlap d'intervalle prioritaire sur le sémantique.

---

## 2. Les **faits gravitent** autour de l'event (clustering event-centrique)

Le motif récurrent : un event sert d'**ancre/hub/centroïde** sur lequel les
faits participants s'attachent, scorés par similarité, puis agrégés.

- **EventKG** : > 690 000 events organisés en **hub** (et non l'entité) ;
  attache les faits par edges typés (acteurs, lieux, temps) avec une
  distant-supervision qui **classe quelles relations sont les plus pertinentes**
  → association *sélective* fact→event.
  [EventKG (arXiv:1905.08794)](https://arxiv.org/abs/1905.08794)
- **ChronoGrapher** : « informed graph traversal » qui **récupère tout le
  sous-graphe event-centrique** (tous les sous-events liés à un event) —
  l'event tire à lui ses faits.
  [ChronoGrapher (SWJ 2025)](https://www.semantic-web-journal.net/system/files/swj3725.pdf)
- **Coréférence d'events cross-document (CDECR)** = *clusteriser* les mentions/
  faits qui réfèrent au même event réel. SOTA en **fusionnant l'encodage de la
  mention avec un résumé-event** comme ancre enrichie (CoNLL F1 86.7 ECB+).
  Reformulable en **retrieval** : une mention-event comme requête → retrouver
  les coréférents via DPR.
  [Synergetic (arXiv:2406.02148)](https://arxiv.org/abs/2406.02148) ·
  [Event Coref Search (arXiv:2210.12654)](https://arxiv.org/abs/2210.12654)
- **Centroïde = saillance** : TDT organise l'info **par event** et suit des
  **centroïdes** ; **MEAD/Radev** score la saillance d'un fait par sa
  **centralité (similarité) au centroïde** de l'event — l'analogue formel le
  plus proche de la « gravitation ».
  [TDT](https://link.springer.com/chapter/10.1007/0-306-47019-5_4) ·
  [MEAD (cs/0005020)](https://arxiv.org/pdf/cs/0005020)
- **EventSum** : structure chaque hub-event avec sous-events + arguments typés
  (time/location/person/org) + relations causales ; les docs sont **rassemblés
  autour de l'event** (retrieval + filtre similarité > 0.5) avant résumé.
  [EventSum (arXiv:2412.11814)](https://arxiv.org/html/2412.11814v2)

**Pattern transférable** : *ancre (hub/centroïde/résumé-event) → score de
similarité → agrégation en edges typés (rôle/temps/lieu/cause)*. C'est
exactement notre `apply_pull` (gravitation géométrique, sans edges stockés).

---

## 3. **Schémas de type d'event** (la typologie récurrente)

Un **type** définit un schéma réutilisable de rôles/slots/sous-events.

- **FrameNet** : un *frame* = schéma d'un type d'event + ses *frame elements*
  (rôles). FEs **core** (slots requis du type) vs **non-core** (Time, Place,
  Manner). Relations frame-à-frame : **Inheritance**, **Subframe** (un event
  complexe se décompose en sous-events — ex. Commerce → Payment + Transfer),
  **Precedes** (ordre temporel). ~1000 frames hiérarchisés.
  [FrameNet](https://en.wikipedia.org/wiki/FrameNet)
- **Scripts (Schank & Abelson)** : séquence stéréotypée d'actions + rôles,
  props, conditions, scènes, issues (script du restaurant). Distinction
  **scripts vs plans vs goals** → les schémas sont activés par, et se
  décomposent vers, des **buts**. Les **MOPs** généralisent autour d'un but
  commun. [Scripts, Plans, Goals (1977)](https://garfield.library.upenn.edu/classics1989/A1989AN57700001.pdf)
- **Ontologies d'events = inventaires de slots par type** : **ACE-2005**
  (8 types / 33 sous-types / 35 rôles, chaque type a un *template* de rôles
  attendus) ; **RAMS** (139 types / 65 rôles, slots remplis cross-phrases) ;
  **MAVEN-Arg** (162 types / 612 rôles, le plus grand inventaire unifié).
  [ERE (NAACL 2015)](https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/naacl2015-light-to-rich-ere.pdf) ·
  [RAMS (arXiv:2104.05919)](https://arxiv.org/pdf/2104.05919) ·
  [MAVEN-Arg (arXiv:2311.09105)](https://arxiv.org/abs/2311.09105)
- **Induction de schéma** (apprendre les templates récurrents du corpus) :
  - **Narrative event chains** (Chambers & Jurafsky 2008) : events partageant
    un protagoniste, + éval *Narrative Cloze*. [ACL 2008](https://aclanthology.org/P08-1090/)
  - **Event Graph Schema** (Li 2020) : deux types reliés par **plusieurs chemins
    médiés par des entités** ; induit par un *Path Language Model*. Schéma =
    **graphe typé**, pas une chaîne. [EMNLP 2020](https://aclanthology.org/2020.emnlp-main.50/)
  - **Temporal Complex Event Schema** (Li 2021) : graphe events+arguments+temps
    +relations ; +23.8 % HITS@1 en prédiction d'event futur.
    [arXiv:2104.06344](https://arxiv.org/abs/2104.06344)
  - **Hierarchical schema induction par LLM** (Li 2023) : *Incremental Prompting
    & Verification* en 3 étapes (squelette → expansion → vérif des relations) ;
    +31.0 % F1 hiérarchique. → schéma = **hiérarchie de sous-events + edges
    temporels**, inductible par LLM. [ACL 2023 (arXiv:2307.01972)](https://arxiv.org/abs/2307.01972)

**Ce que contient concrètement un schéma** : (a) label de type ; (b) inventaire
de **slots/rôles** (core requis vs périphériques Time/Place) ; (c) **sous-events
/scènes** (Subframe/hiérarchie) ; (d) **ordre temporel** (Precedes) ; (e)
**liaisons entité-rôle** (même participant à travers les sous-events). →
*war ⇒ belligérants/pertes = rôles ; fronts/lands = slots ; timeline = edges
temporels ; traités = sous-events.* Exactement ton intuition.

---

## 4. Du schéma aux **sous-questions** (décomposition + slot-filling)

Le schéma agit comme un **plan** qui produit une sous-requête par slot.

- **Question Decomposition for RAG** : décomposer → retrieve par sous-question →
  rerank vs requête originale ⇒ **MRR@10 0.464→0.635 (+36.7 %)** sur
  MultiHop-RAG. *« decomposition expands coverage, reranking restores
  precision »*. **Mais** la décomposition libre **sur-génère** (5 sous-requêtes
  dans 93–99 % des cas vs 2–3 nécessaires) → **un schéma à nombre de slots fixe
  calibre mieux le budget.** [arXiv:2507.00355](https://arxiv.org/abs/2507.00355)
- **Plan-RAG** : plan = **DAG** de sous-requêtes atomiques (parallélise les slots
  indépendants, séquence quand un slot nourrit le suivant).
  [OpenReview](https://openreview.net/forum?id=cUuOKnjVQJ)
- **Slot = une question** : la *template-filling* définit `k+1` slots (1 type +
  k rôles) → **une question par slot**. **QGA-EE** génère une question NL
  *par rôle* depuis l'ontologie (SOTA ACE05) ; **RLQG** raffine la question de
  slot (4 critères : fluence, généralisable, dépendante du contexte,
  indicative). [QGA-EE (arXiv:2307.05567)](https://arxiv.org/abs/2307.05567) ·
  [RLQG (arXiv:2405.10517)](https://arxiv.org/html/2405.10517v2)
- **ASEE** : *récupérer le schéma* (paraphrase + embedding-match BGE-M3/BM25
  parmi des centaines) **puis** extraire conditionné dessus → résout fenêtre de
  contexte + hallucination (benchmark MD-SEE, 300 schémas).
  [ASEE (arXiv:2505.08690)](https://arxiv.org/abs/2505.08690)
- Grounding linguistique : **QUD** (Questions Under Discussion) — chaque unité
  de discours répond à une question implicite ⇒ « schema-to-questions ».
  [QUD survey (arXiv:2502.15573)](https://arxiv.org/html/2502.15573v1)

**Recette** : (1) sélectionner/induire le schéma du type ; (2) une sous-question
par slot (nombre fixe = budget calibré) ; (3) la réaliser en **question NL
contextualisée** (pas un mot-clé) ; (4) émettre en **DAG** (parallèle/séquentiel)
scopé au slot ; (5) retrieve par sous-question + rerank vs requête.

---

## 5. **Opérationnaliser** un RAG event-aware (+ problèmes ouverts)

- **DyG-RAG** = l'exemplaire le plus complet : restructure le texte en
  **Dynamic Event Units** `{phrase, timestamp normalisé, event-id, source}` →
  l'**unité de retrieval est un event time-ancré, pas un chunk**. Typage
  **règle-based** (NER ∨ prédicat de changement d'état ∨ quantité ∨ précision
  temporelle ≥ mois). Graphe : edge entre 2 DEU **ssi entité partagée ET
  `|tᵢ−tⱼ| ≤ Δt`**, poids `sim·exp(−α|tᵢ−tⱼ|)`. **Time-CoT** filtre au scope
  temporel + classe *instantané vs ongoing*. ⇒ **+18.3 % vs GraphRAG** sur
  TimeQA (« events-as-units > chunks-as-units »).
  [DyG-RAG (arXiv:2507.13396)](https://arxiv.org/abs/2507.13396)
- **Schema lookup/induction** : ASEE (lookup) + **Zero-Shot On-the-Fly Schema
  Induction** (génère des docs synthétiques depuis une définition de type →
  schéma sans données) pour les **types inconnus** ; schémas LLM-induits
  *plus complets* que curés humains dans une majorité de cas.
  [Zero-Shot Schema Induction (arXiv:2210.06254)](https://arxiv.org/abs/2210.06254)
- **Scoped slot-filling** : **self-query retriever** (le LLM infère des filtres
  métadonnée — timestamp/type/source — depuis la question) +
  **Robust RAG zero-shot slot filling** (DPR + hard negatives + générateur qui
  marginalise sur l'évidence) pour remplir un slot de type inédit.
  [Robust RAG slot filling (arXiv:2108.13934)](https://arxiv.org/abs/2108.13934)
- **Mémoire épisodique d'agent** : 5 propriétés (long terme, raisonnement
  explicite, single-shot, instance-spécifique, **relations contextuelles
  when/where/why/who**) ; stocker chaque épisode avec métadonnées what/when/
  where/why puis le réinstancier. [Episodic Memory (arXiv:2502.06975)](https://arxiv.org/abs/2502.06975)

**Problèmes ouverts** : (a) **events ouverts/ongoing** sans timestamp terminal →
flags de *validité courante* + décroissance de confiance ; distinguer
clos/ongoing sans mal-ranker les faits périmés (*knowledge drift*) reste non
résolu ; (b) **interférence multi-events** (entité dans plusieurs events
temporellement chevauchants) + raisonnement causal ; (c) **segmentation** du
flux en épisodes (event boundary detection) ; (d) **schéma incomplet / types
nouveaux / schema drift**. [TG-RAG](https://arxiv.org/pdf/2510.13590) ·
[DyG-RAG](https://arxiv.org/html/2507.13396v1)

---

## 6. Blueprint concret pour **MetaCog-Mem**

On a déjà l'essentiel des briques — il s'agit de les assembler en sous-système
event-schéma :

| Brique littérature | Notre équivalent (existant) |
|---|---|
| Event = hub ; faits gravitent (EventKG/centroïde) | **`apply_pull`** (gravitation géométrique edge-free) → tirer les faits source sur un **node-event** |
| Extension temporelle, scope dur τ∉T⇒0 (TG-RAG/TSM) | **tags date déterministes** + **`walk_start(date_tags→restrict_ids)`** (hard-scope) ; right-open = event sans `end` |
| `event:type:*` → schéma de rôles (FrameNet/ACE/MAVEN) | **tags hiérarchiques `event:type:war:slot:*`** + tag glossary/navigator |
| Induction de schéma par LLM (Li 2023) | **refine_tags / un `induce_event_schema(type)`** caché par type (cache comme clue_search) |
| Schéma → sous-questions (QGA-EE/Plan-RAG) | **clue_search** (déjà de l'answer-space) → généraliser en **une sous-question par slot** |
| Slot-filling scopé (self-query/ASEE) | **presearch/scoped** sur `event:type:war:slot:lands` + `restrict_ids` temporel |

**Pipeline proposé** (event-schema walk) :
1. **Détection/typage à l'ingestion** : repérer un event (règle DyG-RAG : NER +
   prédicat de changement d'état + date), créer un **node-event** `event_<id>`
   tag `event:type:war`, **`apply_pull(fact, event_node)`** pour les faits qui
   le mentionnent (gravitation), et stocker l'**intervalle** `[start, end?]`
   (date tags ; `end` absent = ongoing/right-open).
2. **Schéma par type** : `induce_event_schema("war")` (LLM, 3 étapes Li 2023,
   caché) → slots récurrents `{belligerents, lands, locations:{camps,fronts},
   casus_belli, timeline, casualties, treaties}`, matérialisés en sous-tags
   `event:type:war:slot:*`.
3. **THOUGHT** : sur une question d'event, détecter `event + type` → **charger
   le schéma**, reconnaître la contrainte temporelle (intervalle de l'event).
4. **ACTION** : émettre **une sous-question par slot** (QGA-EE), chacune
   **scopée** (a) au tag de slot et (b) à l'intervalle de l'event
   (`restrict_ids`), pour **remplir le schéma en rassemblant les faits qui
   gravitent**. Budget de sous-requêtes = nombre de slots (calibré, pas la
   sur-génération libre).
5. **Synthèse** : agréger les slots remplis (centroïde-event) ; marquer les
   slots vides ; trier par timeline.

**Gain attendu** vs notre answer-space actuel : la décomposition par slot
**calibre le budget** et **couvre exhaustivement** la typologie (vs clues
stochastiques qui ratent une branche) — directement le levier qui manquait sur
les cas obliques (couvrir *toutes* les sous-questions latentes d'un type).

---

## 7. Nouveau **PointKind.EVENT** + système d'**épisodes** (modèle Graphiti / Zep)

**Graphiti** (Zep, [arXiv:2501.13956](https://arxiv.org/html/2501.13956v1)) est l'art
antérieur direct. Son architecture = **3 tiers de sous-graphes** :
- **Episode subgraph** : des **Episodic nodes** = events/messages **bruts**
  horodatés (la *ground truth* haute-fidélité : logs de conversation, JSON,
  snapshots).
- **Semantic entity subgraph** : entités + edges **extraits** des épisodes ; les
  **episodic edges `Ee`** relient un épisode à ses entités extraites.
- **Community subgraph** : clusters/communautés.

Et un **modèle bi-temporel** : chaque entity edge porte **deux axes** —
*event time T* (quand le fait était vrai dans le réel) et *ingestion time T′*
(quand le système l'a appris) — avec des **intervalles de validité explicites
`(t_valid, t_invalid)`** ⇒ **invalidation d'edge** quand un fait est supplanté.

### Mapping sur MetaCog-Mem — un 4ᵉ type de node

Aujourd'hui : `PointKind.FACT / ACTION / THOUGHT`. On ajoute **`PointKind.EVENT`**
comme **hub de 1ʳᵉ classe**, et on adopte la hiérarchie Graphiti (qu'on a déjà
en grande partie) :

| Tier Graphiti | Notre node | État |
|---|---|---|
| **Episodic node** (turn/message brut horodaté) | **EPISODE** = le turn/session brut (nos FACT-turns `[date]…` avec tags date déterministes) | ✅ existe (l'épisode = le turn) |
| episodic edge `Ee` (épisode → entité) | `parents=[episode_id]` + **`apply_pull`** (atomics/entités déjà parentés au turn) | ✅ existe |
| entity / fact extrait | nos **atomic facts** + **entity beacons** | ✅ existe |
| **EVENT hub** (abstraction : « la guerre », « l'adoption de John ») | **`PointKind.EVENT`** *(nouveau)* | ❌ à créer |
| community | **observators / clusters** | ✅ existe |

### Le node EVENT (nouveau) — ce qu'il porte

```
Point(kind=EVENT, id="event_<uuid>")
  tags        : ["event", "event:type:war", "event:type:war:slot:*" …]  # le schéma
  interval    : t_start, t_end?            # right-open si ongoing (cf §1)
  bi-temporal : event_time (date tags) + ingestion_time (t_last_obs)     # cf Graphiti
  gravitation : apply_pull(fact_i, event_node)  pour chaque fait qui le mentionne  # cf §2
```

- **Détection/typage** à l'ingestion (règle DyG-RAG : NER + prédicat de
  changement d'état + date) → crée/dédup un `event_` node, tag `event:type:…`.
- **Gravitation** : `apply_pull` co-localise les faits-épisodes autour du hub —
  notre analogue **géométrique edge-free** des episodic edges `Ee`.
- **Schéma** : `event:type:war` → slots récurrents (§3) → **sous-questions**
  (§4) scopées au slot **et** à l'intervalle (§1).
- **Bi-temporel + invalidation** : un fait supplanté → marquer `t_invalid`
  (relie à notre système de collision/révision `n_revision`/contradiction) ;
  *ongoing* = `t_end` absent (right-open).

### Pourquoi un node dédié (et pas juste un tag)
Un EVENT n'est pas un fait : c'est un **agrégateur** avec sa propre identité,
son intervalle, son schéma, et sa **dynamique de gravitation**. Comme l'Episodic
node de Graphiti est distinct des entités, le node EVENT est distinct des FACT —
il est l'**ancre** que la walk/anchor peut viser (`walk_start(observator_id=` ou
un futur `event_id=`) pour rassembler tout le schéma d'un coup.

**Changements de schéma minimaux** : (1) ajouter `EVENT` à `PointKind` ;
(2) `Memory.ingest_event(type, value, *, source_facts, interval)` (miroir de
`ingest_entity`, avec `apply_pull` sur chaque source) ; (3) `event:type:*` dans
le tag glossary (déjà géré par le navigateur) ; (4) `induce_event_schema(type)`
caché (LLM, Li 2023) ; (5) la walk : THOUGHT détecte l'event+type → ACTION émet
les sous-questions de slot scopées interval+tag. Tout le reste (pull, tags,
date hard-scope, clue_search) est **déjà là**.

---

## Sources (consolidées)

Temporel : [TLEX](https://arxiv.org/html/2406.05265) ·
[Survey TKG](https://arxiv.org/html/2403.04782v1) ·
[TEMT](https://arxiv.org/abs/2309.16357) ·
[TSM](https://arxiv.org/html/2601.07468v1) ·
[Episodic Memory](https://arxiv.org/abs/2502.06975) ·
[TG-RAG](https://arxiv.org/html/2510.13590v1) ·
[STAR-RAG](https://arxiv.org/html/2510.16715v1)

Gravitation : [EventKG](https://arxiv.org/abs/1905.08794) ·
[ChronoGrapher](https://www.semantic-web-journal.net/system/files/swj3725.pdf) ·
[Synergetic CDECR](https://arxiv.org/abs/2406.02148) ·
[WEC](https://aclanthology.org/2021.naacl-main.198/) ·
[Event Coref Search](https://arxiv.org/abs/2210.12654) ·
[TDT](https://link.springer.com/chapter/10.1007/0-306-47019-5_4) ·
[MEAD](https://arxiv.org/pdf/cs/0005020) ·
[Event graphs IR](https://www.sciencedirect.com/science/article/abs/pii/S0957417414001985) ·
[EventSum](https://arxiv.org/html/2412.11814v2)

Schémas : [FrameNet](https://en.wikipedia.org/wiki/FrameNet) ·
[Schank & Abelson](https://garfield.library.upenn.edu/classics1989/A1989AN57700001.pdf) ·
[ERE](https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/naacl2015-light-to-rich-ere.pdf) ·
[RAMS](https://arxiv.org/pdf/2104.05919) ·
[MAVEN-Arg](https://arxiv.org/abs/2311.09105) ·
[Narrative Event Chains](https://aclanthology.org/P08-1090/) ·
[Event Graph Schema](https://aclanthology.org/2020.emnlp-main.50/) ·
[Temporal Complex Event Schema](https://arxiv.org/abs/2104.06344) ·
[Double Graph Autoencoders](https://aclanthology.org/2022.naacl-main.147/) ·
[Hierarchical Schema Induction](https://arxiv.org/abs/2307.01972)

Sous-questions : [Question Decomposition RAG](https://arxiv.org/abs/2507.00355) ·
[Plan-RAG](https://openreview.net/forum?id=cUuOKnjVQJ) ·
[QGA-EE](https://arxiv.org/abs/2307.05567) ·
[RLQG](https://arxiv.org/html/2405.10517v2) ·
[ASEE](https://arxiv.org/abs/2505.08690) ·
[Low-Resource Template](https://arxiv.org/abs/2205.12643) ·
[QUD survey](https://arxiv.org/html/2502.15573v1)

Opérationnalisation : [DyG-RAG](https://arxiv.org/abs/2507.13396) ·
[Zero-Shot Schema Induction](https://arxiv.org/abs/2210.06254) ·
[Robust RAG slot filling](https://arxiv.org/abs/2108.13934) ·
[Self-Query Retrievers](https://medium.com/@lorevanoudenhove/enhancing-rag-performance-with-metadata-the-power-of-self-query-retrievers-e29d4eecdb73)

Node EVENT / épisodes / bi-temporel :
[Zep / Graphiti (arXiv:2501.13956)](https://arxiv.org/html/2501.13956v1) ·
[Graphiti — Neo4j blog](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
