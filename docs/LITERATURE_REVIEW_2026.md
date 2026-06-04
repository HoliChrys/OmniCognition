# MetaCog-Mem — Revue de littérature & feuille de route (mai 2026)

Synthèse de 5 recherches parallèles (~40 sources) sur comment améliorer
MetaCog-Mem sur LoCoMo. TL;DR : notre design edge-free est **aligné avec
l'évidence 2025-2026** ; les gros leviers manquants sont *reranking*,
*Chain-of-Note*, *consolidation hiérarchique* et *scaffolding temporel*.

---

## 0. Recadrage métrique (le plus important)

- **Le benchmark a changé de métrique.** LoCoMo 2024 = token-F1 (HeLa-Mem
  multi-hop ~0.40, GPT-4 overall ~32). Depuis 2025 quasi tous les systèmes
  reportent un **LLM-as-Judge** sur 1 540 QA (4 catégories, pas 5). Les
  J-scores de 88-92 % (ByteRover, Mem0-2026, MemMachine) **ne sont pas
  comparables** à notre F1 0.22.
- **Bug F1 trouvé & corrigé** : notre `f1_score` ne strippait pas les
  articles (a/an/the) alors que le `normalize_answer` officiel SQuAD le
  fait. "a transgender woman" vs "transgender woman" était plafonné à 0.8.
  ✅ Corrigé (commit `8b327f1`).
- **Plafond full-context = 87.5** (Memori) : stuffer toute la conversation
  dans un long-context LLM bat la plupart des systèmes mémoire. Notre
  valeur ajoutée doit être le coût/passage à l'échelle, pas juste le score.

## 1. Leaderboard LoCoMo (J-score %, 2025-2026)

| Système | Single | Multi | Temp | Open | Overall | LLM | Technique clé |
|---|---|---|---|---|---|---|---|
| Mem0 | 67 | 51 | 56 | 73 | 67 | 4o-mini | fait atomique + dense |
| Mem0-Graph | 66 | 47 | 58 | 76 | 68 | 4o-mini | +triples (gain ~+2pp) |
| Zep/Graphiti | 62 | 41 | 49 | 77 | 66 | 4o-mini | KG temporel |
| MIRIX | — | — | — | — | **85.4** | — | 6 types mémoire + multi-agent |
| MemMachine v0.2 | 94 | 90 | 89 | 75 | ~88 | 4.1-mini | +Cohere rerank |
| Mem0 (2026) | — | — | — | — | 92.5 | 4o-mini | ADD-only + BM25+dense+entité |
| ByteRover 2.0 | 95 | 85 | 94 | 77 | 92 | Gemini-3 | Context Tree hiérarchique |

⚠️ Presque tout vient de blogs vendeurs (sauf Mem0 et EMem sur arXiv).
Confounds majeurs : modèle de base (Gemini-3 vs 4o-mini = +5-10pp),
juge LLM, méthodo contestée (Mem0 vs Letta).

## 2. Edge-free vs graphe : **garder edge-free**

- Mem0+graph ne gagne que **+2pp** sur Mem0 vectoriel. GraphRAG coûte
  10-100× en tokens d'indexation, perd 4-18 % sur narratif (NovelQA).
- Désambiguïsation d'entités cascade : 85 % × 5 hops ≈ 44 % fiables.
- **Ablation HeLa-Mem** (arXiv 2604.16839, GPT-4o-mini, F1 34.74 %) :
  - −spreading activation (edges Hebbian) : **−2.55pp**
  - −Reflective Memory Agent : **−4.87pp** ← le vrai levier
  - −adaptive forgetting : −0.46pp
- **Conclusion** : la consolidation réflective (notre THOUGHT) compte 2×
  plus que les edges. RAPTOR (hiérarchie edge-free) récupère l'essentiel
  du bénéfice multi-hop des graphes.

### Edges bon marché à envisager (sans appel LLM/tour)
1. **Edges temporels next-turn** — gratuit, déterministe (`t_prev`/`t_next`).
2. **Co-activation Hebbian** dérivée du spreading déjà calculé.
3. **Chaînes typées FACT→ACTION→THOUGHT** par proximité temporelle.

À éviter : extraction entité-relation (coût $/tour, hallucine, +2-7pp seulement).

## 3. Retrieval — top 3 manquants

| Rang | Technique | Gain | Coût | Difficulté |
|---|---|---|---|---|
| 1 | **Cross-encoder reranker** (bge-reranker-v2-m3) | +5-15 nDCG | 1 modèle/appel | faible (étage final post-RRF) |
| 2 | **Query decomposition** multi-hop | +37 % MRR, +12 F1 | 1 appel LLM | faible (pré-retrieval) |
| 3 | **RAPTOR** (arbre de résumés par session) | +20 % QuALITY | résumé 1×/session | moyen |

À déprioriser : HyDE (instable sur dialogue), ColBERT (infra lourde),
SPLADE (rebuild index). Bonus temporel : prior de récence + extraction
de dates par tour → cible la catégorie temporelle.

## 4. Multi-hop — 3 changements MetaWalker

1. **IRCoT** : re-ancrer sur la dernière phrase du THOUGHT (pas les
   keywords extraits) + émettre 1 sous-question explicite (Self-Ask).
   Stop quand "no follow-up needed". *Cheapest predicted lift.*
2. **Chain-of-Note** : entre retrieval FACT et THOUGHT, 1 appel notant
   chaque fait {relevant/partial/irrelevant/contradicts}. Drop irrelevant,
   surface contradicts. Corrige le #1 failure mode (THOUGHT pollué par
   tours adjacents-mais-faux) + donne un vrai signal de breadth-pivot
   (pivot quand relevant_count==0, au lieu de l'heuristique σ).
3. **Beam width-2 + gate FLARE** : garder top-2 chaînes, ne brancher que
   sur les hops où le LM est incertain. Tree-of-Thoughts-lite sans payer
   2× retrieval partout. Cible le dual-path retriever de HeLa-Mem.

Bonus : Plan-and-Solve pré-passe (checklist de faits à chercher) pour
donner une cible concrète au breadth-pivot.

## 5. Génération terse — 3 recos McpMetaAgent

1. **Strict tool use Anthropic** : un seul outil `final_answer(answer: str)`,
   `tool_choice` forcé au dernier tour, `enum` pour classes fermées. C'est
   le constrained-decoding réel sur l'API Anthropic (grammaire au décodage).
   Tue 80 % du tail verbeux. `terse()` devient filet de sécurité.
2. ✅ **Article-stripping f1_score** (fait).
3. **Few-shot 6-10 paires (Q, A-terse)** dans AGENT_SYSTEM, une par
   catégorie + `max_tokens=24` + stop seqs `["\n", ".", " because", " who"]`
   sur le tour final forcé.

Tier 2 : extracteur 2e passe (gate `len>5 mots`), mode span/extractif
pour single-hop (extractif ~92 F1 vs génératif ~82). Tier 3 (skip) :
DPO/Outlines = modèle self-hosted, perd Haiku.

---

## Feuille de route recommandée (ROI décroissant)

| Priorité | Action | Effort | Gain attendu |
|---|---|---|---|
| P0 ✅ | f1 article-stripping | fait | aligne métrique |
| P0 ✅ | BM25 keyword-indexé + stemming | fait | recall multi-hop |
| P1 | `final_answer` strict tool use | faible | tue tail verbeux |
| P1 | Chain-of-Note relevance pass | moyen | #1 failure mode + pivot signal |
| P1 | Cross-encoder reranker post-RRF | faible | +5-15 nDCG |
| P2 | IRCoT sous-question re-anchoring | faible | multi-hop compositionnel |
| P2 | Edges temporels next-turn | faible | catégorie temporelle |
| P2 | Few-shot + max_tokens=24 | faible | terseness |
| P3 | RAPTOR arbre de résumés | moyen | granularité retrieval |
| P3 | Beam width-2 + FLARE gate | élevé | exploration ciblée |

**Principe directeur** : MetaCog-Mem reste hyperparameter-free et edge-free.
Chain-of-Note + reranker + IRCoT sont tous *inference-time*, sans
entraînement, et composables. Gain empilé estimé : +10-20 points de recall
sur un baseline LoCoMo, sans toucher au coeur géométrique.
