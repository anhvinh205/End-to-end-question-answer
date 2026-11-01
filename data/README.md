# Data

Nothing large is committed. This folder holds:

| Path | In git? | What it is |
|---|---|---|
| `raw/Natural-Questions-Base.csv` | no | Input. Natural Questions export with columns `question`, `long_answers`, `short_answers` (307,373 rows, ~330 MB). |
| `processed/{train,validation,test}.jsonl` | no | Output of `make data`: cleaned, aligned records, split by passage. |
| `processed/audit.json` | no | Where every CSV row went (also copied to `reports/data_audit.json`, which *is* committed). |
| `sample/nq_sample.jsonl` | **yes** | 300 answerable validation records. Lets `make api` / `make ui` work right after `git clone`. |

## Use the Base file, not the Filtered one

Both CSV variants share the same columns. They are not equivalent:

| | `Natural-Questions-Base.csv` | `Natural-Questions-Filtered.csv` |
|---|---|---|
| Rows | 307,373 | 86,212 |
| Rows with a passage and a short answer | 106,926 | 86,212 (86,208 of them also exist in Base) |
| Passage text | NQ tokens with HTML (`<P>`, `<Td>` ...), faithful | cleaned, but **18.1 % of the digit tokens are gone** (61,525 of 339,051 in the overlapping rows) and some names are altered |
| Short answer found verbatim in the passage | 90.2 % | 66.7 % (89.5 % ignoring case) |

Dropped digits break exactly the questions that have numeric answers (dates, scores, ages), so
`configs/config.yaml` points at the Base file. The loader also reads the Filtered file (same
columns), but a QA benchmark built on it measures a corrupted corpus.

## What `make data` does to the Base file

`scripts/prepare_data.py` -> `nq_qa/data/nq_csv.py`

1. **Rows without a passage** (155,225, 50.5 %) are skipped: there is no context to read.
2. **Tables are skipped** (`<Table>`, `<Tr>`: 31,238 passages): flattened cells are not prose.
   Paragraphs and lists are kept (`dataset.context_types`).
3. **HTML tokens are stripped** and the passage is rebuilt as natural text (`firm 's` -> `firm's`,
   `7 : 00` -> `7:00`), **recording the character offset of every token**.
4. **The short answer is located at token level** (exact, then case-insensitive) and mapped to
   character offsets through those token offsets. 9 % of the answers are several spans joined by
   `", "`; they are split and every span that can be found is kept, SQuAD-style (`answers.text` is a
   list). Naive `short in long` string matching would silently lose this 10 %.
5. **Length filters**: passages of 10-350 words, answers of at most 20 words.
6. **Rows with a passage but no short answer** (31,920 after filtering) are kept as natural
   *negative* examples (`has_answer: false`).
7. **Group-aware split by passage hash** (90 / 5 / 5 %). Several questions share one Wikipedia
   paragraph; splitting by question would leak paragraphs from train into validation.

Result of the full run (`reports/data_audit.json`):

| Split | Records | Answerable |
|---|---:|---:|
| train | 103,497 | 74,745 |
| validation | 5,766 | 4,135 |
| test | 5,532 | 3,995 |

Every one of the 85,622 short answers that survived the passage filters could be located
(alignment rate 100 %); 2,747 were dropped for being longer than 20 words.

## Record format

```json
{
  "id": "nq-1234",
  "question": "who played the incredible hulk in the 70s",
  "context": "The Incredible Hulk is an American television series ... Lou Ferrigno as the Hulk ...",
  "passage_id": "3f2a9c1d5b7e8a40",
  "answers": {"text": ["Lou Ferrigno"], "answer_start": [197]},
  "has_answer": true,
  "context_type": "P"
}
```

`context[answer_start : answer_start + len(text)] == text` is asserted in the tests for the bundled sample.

## Licence and attribution

Natural Questions: Kwiatkowski et al., *Natural Questions: a Benchmark for Question Answering
Research*, TACL 2019. The annotations and the Wikipedia text are released under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The bundled
`sample/nq_sample.jsonl` is a small excerpt and carries the same licence; the code in this repository is MIT.
