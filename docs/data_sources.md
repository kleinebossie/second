# Data Source Licenses and Distribution Decisions

> [!NOTE]
> This document records project decisions for the Second project.
> This document is not legal advice.
> If you need legal advice, consult an attorney.

---

## 1. Overview and Summary

The Second project uses chess games, chess puzzles, engine evaluations, and human commentary.
The project trains language models to explain chess moves clearly.
We verified the legal terms and licenses for all five primary data sources from the project plan.
This document defines how we handle each data source.

We divide distribution into two categories:
1. **Public processed data:** When licenses permit public redistribution, we publish clean dataset files to Hugging Face (`second-chess/second-data-1.0`).
2. **Rebuild script only:** We do not publish raw or processed dataset files. We publish only a Python script. The script downloads and processes the data on the local computer of the user.

### Summary Table

| Data Source | Source URL / Citation | Exact License | Redistribution Rights | Project Decision for Second |
|---|---|---|---|---|
| Lichess Game Database | `https://database.lichess.org/#standard_games` | CC0 1.0 Universal | Full public domain rights worldwide. | Publish processed data in `second-chess/second-data-1.0`. |
| Lichess Puzzles Database | `https://database.lichess.org/#puzzles` | CC0 1.0 Universal | Full public domain rights worldwide. | Publish processed data in `second-chess/second-data-1.0`. |
| Lichess Studies | `https://lichess.org/study`<br>`https://lichess.org/terms-of-service` | Lichess Terms of Service (Author copyright) | Third parties receive no redistribution rights for text comments. | Publish rebuild script only. Publish data only if authors set open licenses. |
| GameKnot Annotated Games | `https://gameknot.com/best-annotated-games.pl`<br>`https://gameknot.com/pg/pol_eula.htm` | Proprietary (EULA Section 8, All Rights Reserved) | No redistribution rights. GameKnot owns user submissions. | Train on it. Publish rebuild script only. Do not publish raw or processed text. |
| MATE Dataset | `https://huggingface.co/datasets/OutFlankShu/MATE_DATASET`<br>Wang et al. (NAACL 2025) | MIT License | Full rights with attribution and license notice. | Publish processed data in `second-chess/second-data-1.0` with MIT notice. |

---

## 2. Detailed License Findings

### 2.1 Lichess Open Game Database

- **Source URL:** [https://database.lichess.org/#standard_games](https://database.lichess.org/#standard_games)
- **Verified License:** Creative Commons CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.
- **Redistribution Rights:**
  Lichess waives all copyright and neighboring rights to these database exports worldwide.
  Anyone can copy, modify, publish, and distribute the game records for any purpose.
  Commercial use does not require permission.
  Attribution is appreciated but not legally required under CC0.
- **Decision for Second:**
  We will extract game positions, move metadata, and Stockfish evaluations from this archive.
  We will publish all processed training pairs in `second-chess/second-data-1.0`.
  The dataset will use the Apache 2.0 license.

### 2.2 Lichess Puzzles Database

- **Source URL:** [https://database.lichess.org/#puzzles](https://database.lichess.org/#puzzles)
- **Verified License:** Creative Commons CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.
- **Redistribution Rights:**
  Lichess publishes puzzle exports (such as `lichess_db_puzzle.csv.zst`) under CC0 1.0.
  Anyone can download, change, and redistribute the puzzle database freely.
  Commercial projects can use these puzzle records without fees or restrictions.
- **Decision for Second:**
  We will convert puzzle positions, themes, and solution lines into factual prompts.
  We will publish the processed puzzle dataset in `second-chess/second-data-1.0`.
  The dataset will use the Apache 2.0 license.

### 2.3 Lichess Studies

- **Source URL:** [https://lichess.org/study](https://lichess.org/study) and [https://lichess.org/terms-of-service](https://lichess.org/terms-of-service)
- **Verified License:** Lichess Terms of Service (User Content and Permissions).
- **Redistribution Rights:**
  Under the Lichess Terms of Service, users retain copyright in content they create.
  When users upload annotations and commentary, they grant Lichess a non-exclusive license.
  This grant allows Lichess to host and display the content on the website.
  This grant does not give third parties permission to redistribute user commentary.
  The Lichess website platform uses the AGPLv3 software license.
  The AGPLv3 software license does not cover user-created annotations.
  Standard chess move notation consists of objective facts and does not receive copyright protection.
  However, human prose commentary represents original creative expression and receives copyright protection.
- **Decision for Second:**
  We will not publish scraped study commentary in `second-chess/second-data-1.0`.
  If an individual study creator explicitly states an open license (such as CC0 or CC-BY), we can include that study.
  For all other studies, we will publish only a rebuild script (`data/sources/lichess_studies.py`).
  The rebuild script will fetch and process studies directly on the computer of the user.

### 2.4 GameKnot Annotated Games

- **Source URL:** [https://gameknot.com/best-annotated-games.pl](https://gameknot.com/best-annotated-games.pl), [https://gameknot.com/pg/pol_eula.htm](https://gameknot.com/pg/pol_eula.htm), and [https://gameknot.com/pg/pol_community.htm](https://gameknot.com/pg/pol_community.htm)
- **Verified License:** Proprietary. GameKnot End User License Agreement (EULA) and All Rights Reserved.
- **Redistribution Rights:**
  Section 8 of the GameKnot EULA governs user-submitted content.
  Users irrevocably assign all copyright and ownership in their content exclusively to GameKnot.
  GameKnot marks website pages with "All rights reserved".
  GameKnot does not provide an open license to the public for commentary text.
  Section 10 of the GameKnot Community Standards forbids automated bots and scrapers.
  Therefore, third parties have no legal right to redistribute GameKnot commentary corpora.
- **Decision for Second:**
  We will not publish raw GameKnot annotations or processed commentary in `second-chess/second-data-1.0`.
  We will publish only a rebuild script (`data/sources/gameknot.py`).
  If a user wants to train on GameKnot data, the user must run the rebuild script locally.
  This approach protects the Second project repository against copyright infringement claims.

### 2.5 MATE Dataset ("Move on strAtegy and Tactics datasEt")

- **Source URL:** [https://huggingface.co/datasets/OutFlankShu/MATE_DATASET](https://huggingface.co/datasets/OutFlankShu/MATE_DATASET)
- **Project Site:** [https://mate-chess.github.io/](https://mate-chess.github.io/)
- **Academic Citation:** Wang et al., "Explore the Reasoning Capability of LLMs in the Chess Testbed", NAACL 2025.
- **Verified License:** MIT License.
- **Redistribution Rights:**
  The authors release `OutFlankShu/MATE_DATASET` under the permissive MIT License.
  The MIT License permits anyone to copy, modify, publish, distribute, and sell copies.
  If you redistribute the dataset or substantial portions, you must include the original copyright notice and license notice.
- **Decision for Second:**
  We will process MATE entries into our fact-based instruction prompt format.
  We will publish the processed data in `second-chess/second-data-1.0`.
  We will include the original MIT copyright notice in the dataset card and documentation.
- **Note on Related Academic Datasets:**
  A related dataset comes from Jhamtani et al. (ACL 2018), "Learning to Generate Move-by-Move Commentary for Chess Games from Large-Scale Social Forum Data".
  The repository `harsh19/ChessCommentaryGeneration` contains move-commentary pairs collected from GameKnot.
  The repository does not contain an explicit open-source license file.
  If the Second project uses data from the Jhamtani corpus, we will treat that data like GameKnot data.
  We will not publish that text in `second-chess/second-data-1.0`.
  We will only provide a rebuild script to process it locally.

---

## 3. Summary of Distribution Pipeline

| Data Pipeline Target | Allowed Sources | Excluded Sources |
|---|---|---|
| `second-chess/second-data-1.0` (Hugging Face) | • Lichess Game Database (CC0)<br>• Lichess Puzzles (CC0)<br>• MATE Dataset (MIT)<br>• Lichess Studies with explicit open licenses | • General Lichess Studies (All Rights Reserved)<br>• GameKnot Commentary (Proprietary)<br>• Scraped GameKnot Academic Corpora |
| Rebuild Scripts (`data/sources/`) | • All sources with functional scrapers or API download scripts | None (scripts execute locally under user responsibility) |
