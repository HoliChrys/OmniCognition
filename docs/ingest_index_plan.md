# Plan : indexation à l'ingestion (Document Card)

**Principe directeur** (validé empiriquement par le tone-reading, q0459 22/23 &
q0424 25/25 à ~coût batch) : *toute lecture du document migre à l'ingestion ;
la query ne paie que ce qui dépend d'elle.*

## État mesuré (avant)

- **Ingestion** : jusqu'à 5 appels LLM SÉPARÉS par doc — keywords
  (`extractor.extract`), entités (`_spawn_entities`), atomes (`_spawn_atomics`),
  event (`_spawn_events`), tone/gloss (`_spawn_tone`).
- **Query, LLM** : clues, HyDE, chain_of_note, meta_thought, generate_action,
  provisional_answer (keepup), distill_proposition (caché), juge batch +
  seconde opinion.
- **Query, CPU** : `search_nodes` re-tokenise tout le pool À CHAQUE appel
  (×modes×rounds) ; `bm25_score` recalcule les IDF en re-scannant tous les docs
  par query ; `k_nearest`/spreading scannent tous les points en cosine par stage.

## Phases (ordonnées par ROI)

### Phase 1 — Document Card unifiée  [fondation] — ✅ VALIDÉE

Verdict (q0459/q0424) : loop 22/23 & 25/25, context 13/23 & 21/25 — les deux
canaux intacts. AMENDEMENT validé empiriquement : les EVENTS restent à
l'extracteur DÉDIÉ (la lecture event intégrée à la card était bien plus faible,
context 13→2 ; une lecture courte spécialisée gagne) — la card consolide
keywords/entités/tone/gloss/stance/questions, ses events ne sont qu'un fallback.
Coût : 2 appels/doc (card+events) aujourd'hui, MAIS stance+questions des
phases 3-4 déjà payés → 2 vs 5 à terme.
UN appel LLM structuré par doc à l'ingestion, retournant :
`{keywords, entities, atoms, event{type,name,dates}, tone, intended_gloss,
stance{position, cibles}, questions[2-3]}`.
- Nouveau `metacog/doc_card.py` : extracteur caché, failure-safe, **ne cache
  jamais un résultat vide** ; fallback PAR CHAMP vers les extracteurs existants
  si l'appel combiné échoue.
- `Memory(doc_card=True)` : un `_spawn_card` remplace les 5 hooks (les hooks
  existants restent pour compat / opt-outs individuels).
- Gain : ÷4-5 le coût LLM d'ingestion + une lecture COHÉRENTE unique.
- Tests : fake déterministe ; parité avec les spawns individuels ; never-cache-empty.

### Phase 2 — Index inversé + BM25 incrémental  [CPU, zéro risque] — ✅ VALIDÉE
`metacog/text_index.py` : token-sets bruts + docs BM25 stemmés + postings,
memoizés au premier toucher (compute-on-miss : auto-réparant pour les points
créés hors ingest ; memo re-clé à la mutation des keywords — la card les
réécrit). Équivalence byte-for-byte testée contre les scans naïfs ; ~5× sur le
chemin sim (60 appels/400 docs : 165→34 ms). Les TAGS restent scannés live
(ils mutent : event:in, tone:*, valid:until). Non picklé — reset au `load()`.

### Phase 3 — Stance card générale  [recall + coût judge] — ✅ VALIDÉE (tests)
Depuis la card de Phase 1 : un gloss de POSITION pour TOUS les docs (plain
inclus). Le juge devient un entonnoir : (a) filtre embedding
proposition↔stance, permissif ; (b) batch LLM sur la bande ambiguë ;
(c) seconde opinion per-item sur les seuls rejets (existant). Recall-first à
chaque étage (un accept n'est jamais renversé).

### Phase 4 — doc2query  [recall oblique] — ✅ VALIDÉE (tests)
Depuis la card : 2-3 « questions auxquelles ce doc répond » (formulations
obliques incluses), embeddées. La recherche sémantique score
`max(doc, gloss, questions)`. Attaque le NEEDS_SEMANTIC résiduel.
- Tests : match d'une question token-disjointe du contenu.

### Phase 5 — Cache du seuil géométrique HYBRIDE  [latence ; re-périmétré] — ✅ VALIDÉE (tests)

ÉTUDE DE TERRAIN (re-périmétrage motivé) :
- Le coût dominant n'est PAS le kNN par point : c'est `geometric_spread` qui
  recalcule **O(n²) distances pairwise À CHAQUE APPEL** (80k computes pour
  400 pts) uniquement pour dériver le seuil émergent median−σ — appelé par
  `retrieve_hybrid(use_spreading)` et `_reflective_context` (chaque thought).
- L'horloge avance à chaque opération (`_now()` += 1) et le decay 1/(1+Δt)
  fait dériver les embeddings effectifs ENTRE deux appels même sans pull → un
  cache byte-exact inter-appels est impossible par principe.
- Les scans query→points (walk per-stage, `k_nearest`) sont incachables
  (l'embedding de query change à chaque stage).

DESIGN (hybride, fallback = recalcul exact) :
1. `geometry.GEO_EPOCH` : compteur module incrémenté par `apply_pull` (toute
   mutation de deltas). Toute modification STRUCTURELLE invalide.
2. Cache du SEUIL : clé = (n, hash(tuple des ids du sous-ensemble), GEO_EPOCH).
   Hit → réutilise le seuil (saute le O(n²)) ; miss (pull/ingest/retrait/
   sous-ensemble différent) → **recalcul exact** (= comportement actuel, la
   justesse ne se dégrade jamais sur changement structurel).
3. Approximation ASSUMÉE et bornée : entre deux hits, seule la dérive de
   PUR DECAY (uniforme, lente, sur un statistique robuste median−σ) est
   ignorée. Tout pull/ingest force l'exact.
4. `sleep()` et `load()` vident le cache (rebuild).
5. Le reste de geometric_spread (embs O(n) + scan seeds×n) reste exact et
   recalculé à chaque appel — linéaire, pas un goulot.

Tests : hit→résultat identique sur manifold figé (t explicite) ; pull → epoch
bump → recalcul ; ingest → clé différente → recalcul ; smoke perf (appels
répétés ~n× plus rapides).

## Validation par phase

Après chaque phase : suite complète verte + autopsies q0459/q0424 (recall ≥
22/23 et 25/25, pas de régression) ; Phase 2 ajoute un timing avant/après.
Invariants du repo respectés : hyperparameter-free (les tailles dérivées des
données ou constantes documentées), anti-laundering (cards = GENERATOR via
apply_pull), never-cache-empty, rebuild au load().

## Reste à la query (irréductiblement query-dépendant)

distill_proposition (caché), chain_of_note / thoughts / actions du walk,
match de stance final (seconde opinion), provisional answers du keepup.
