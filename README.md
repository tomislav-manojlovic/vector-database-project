# Qdrant pretraga i analiza STL-10 slika

## Pregled projekta

Projekat povezuje raÄunarski vid i vektorske baze podataka. Za slike iz celog STL-10 dataseta generiÅ¡u se CLIP embedding vektori, koji se zatim Äuvaju i indeksiraju u Qdrantu. Sistem podrÅ¾ava pretragu sliÄnih slika, analizu klasifikacionih greÅ¡aka, pronalaÅ¾enje duplikata i lokalni veb interfejs.

Implementirana su tri nivoa projekta:

1. pretraga vizuelno sliÄnih slika;
2. klasifikacija weighted k-NN metodom i analiza greÅ¡aka;
3. pronalaÅ¾enje pribliÅ¾nih duplikata i pravljenje oÄiÅ¡Ä‡ene kopije dataseta.

CLIP model `openai/clip-vit-base-patch32` predstavlja svaku sliku L2-normalizovanim vektorom dimenzije 512. U Qdrantu se koristi cosine metrika.

## Dataset

Koristi se ceo STL-10 dataset od **113.000 slika**:

| Split | Broj slika | Labele |
|---|---:|---|
| `train` | 5.000 | 10 STL-10 klasa |
| `test` | 8.000 | 10 STL-10 klasa |
| `unlabeled` | 100.000 | bez poznate klase |
| **Ukupno** | **113.000** | |

Deset klasa su: `airplane`, `bird`, `car`, `cat`, `deer`, `dog`, `horse`, `monkey`, `ship` i `truck`.

Slike iz unlabeled skupa ostaju u bazi jer doprinose realnoj veliÄini kolekcije i formiranju HNSW indeksa. U metapodacima imaju:

- `label="unlabeled"`;
- `label_id=-1`;
- `is_labeled=False`;
- `split="unlabeled"`.

Skripta `scripts/prepare_dataset.py` automatski preuzima STL-10 pomoÄ‡u `torchvision`, izvozi slike i pravi:

- `data/metadata.csv` â€“ metapodatke za svih 113.000 slika;
- `data/metadata_sample.csv` â€“ mali uzorak metapodataka;
- `data/images/stl10/` â€“ lokalno izvezene slike koje se ne Äuvaju na Gitu.

Dataset nije potrebno ruÄno preuzimati.

## Arhitektura

```text
STL-10: train + test + unlabeled
                |
                v
       JPEG slike i metadata.csv
                |
                v
        CLIP embedding, 512D
                |
                v
          L2 normalizacija
                |
        +-------+-------+
        |               |
        v               v
   NumPy exact      Qdrant Cosine
                        |
                 HNSW indeks + filteri
                        |
          +-------------+-------------+
          |             |             |
     sliÄne slike  analiza greÅ¡aka  duplikati
```

## Embedding pipeline

`src/generate_embeddings.py` uÄitava slike u batch-evima, prosleÄ‘uje ih CLIP modelu i Äuva rezultate u `data/embeddings/`:

- `embeddings.npy` â€“ matrica oblika pribliÅ¾no `113000 x 512`;
- `embeddings_metadata.csv` â€“ ID, putanja, labela, split, `is_labeled`, indeks embeddinga i status;
- `embedding_config.json` â€“ koriÅ¡Ä‡eni model i konfiguracija embeddinga.

Skripta podrÅ¾ava CPU i CUDA. Opcija `--device auto` automatski bira GPU ako CUDA verzija PyTorch-a i odgovarajuÄ‡i NVIDIA drajver postoje.

## Qdrant kolekcija

Docker Compose pokreÄ‡e `qdrant/qdrant:v1.18.2`. Kolekcija ima sledeÄ‡u konfiguraciju:

| PodeÅ¡avanje | Vrednost |
|---|---|
| Kolekcija | `stl10_clip_images` |
| Vektorska dimenzija | 512 |
| Metrika | Cosine |
| Broj pointova | pribliÅ¾no 113.000 |
| Payload | `id`, `image_path`, `label`, `label_id`, `is_labeled`, `split` |
| Payload indeksi | `label`, `split`, `is_labeled` |
| Import batch | 500 |
| Paralelni import radnici | 4 |

Velika kolekcija prelazi Qdrantov prag za indeksiranje, pa se umesto obiÄnog full scan-a formira HNSW indeks.

## Exact i HNSW pretraga

Postoje dva reÅ¾ima pretrage:

- **exact** proverava sve odgovarajuÄ‡e vektore i koristi se kao referentni rezultat;
- **HNSW** koristi pribliÅ¾ni indeks i podrazumevani je reÅ¾im u analizama i UI serveru.

Za HNSW pretragu vaÅ¾ni su:

- `hnsw_ef` â€“ broj kandidata razmatranih tokom upita; veÄ‡a vrednost obiÄno poveÄ‡ava recall, ali moÅ¾e poveÄ‡ati vreme pretrage;
- `m` â€“ broj veza Ävora u HNSW grafu; veÄ‡a vrednost pravi guÅ¡Ä‡i indeks, zauzima viÅ¡e memorije i moÅ¾e poboljÅ¡ati recall;
- `recall@k` â€“ udeo taÄnih top-k suseda koje je pribliÅ¾na pretraga pronaÅ¡la u odnosu na exact rezultat.

Skripte `07_error_analysis.py` i `08_dataset_cleaning.py` koriste HNSW sa `hnsw_ef=64`, dok opcija `--exact` ukljuÄuje egzaktnu pretragu. Komande `compare-backends` namerno koriste exact reÅ¾im radi poreÄ‘enja sa NumPy backendom.

## Nivo 1: pretraga sliÄnih slika

`src/06_queries.py` podrÅ¾ava dohvatanje pointa, payload filtere, pretragu sliÄnosti i kontrolisane CRUD operacije.

Kod komande `similar` upitna slika se uklanja iz rezultata, tako da se ne prikazuje kao sopstveni sused sa score vrednoÅ¡Ä‡u 1.0. Isto ponaÅ¡anje koristi UI server.

```powershell
python .\src\06_queries.py similar 1 --top-k 5
python .\src\06_queries.py similar 1 --top-k 5 --label bird
```

## Nivo 2: analiza greÅ¡aka

`src/07_error_analysis.py` koristi weighted k-NN nad CLIP vektorima. Sama query slika se iskljuÄuje, a cosine score suseda koristi se kao teÅ¾ina glasa klase.

PoÅ¡to 100.000 unlabeled slika nema stvarnu klasu, analiza taÄnosti radi samo nad 13.000 oznaÄenih slika. Radi brÅ¾eg izvrÅ¡avanja podrazumevano se bira reproduktivan uzorak od 1.000 query slika, dok se susedi traÅ¾e meÄ‘u svih 13.000 oznaÄenih slika. Ovo je leave-one-out analiza CLIP prostora, a ne standardna evaluacija zasebno treniranog klasifikatora.

Primer dobijenog rezultata za 1.000 upita:

- 990 taÄnih predikcija;
- 10 greÅ¡aka;
- taÄnost 99,00%;
- dijagnoze: `ambiguous_or_outlier`, `boundary_case` i `class_confusion`.

IzveÅ¡taji se Äuvaju u `reports/error_analysis/` kao HTML, JSON i CSV fajlovi.

```powershell
python .\src\07_error_analysis.py analyze
python .\src\07_error_analysis.py analyze --max-images 2000
python .\src\07_error_analysis.py analyze --max-images 0
python .\src\07_error_analysis.py analyze --exact
```

Vrednost `--max-images 0` analizira svih 13.000 oznaÄenih slika i zato traje znatno duÅ¾e.

## Nivo 3: pronalaÅ¾enje duplikata

`src/08_dataset_cleaning.py` traÅ¾i veoma sliÄne parove i grupiÅ¡e ih u povezane komponente. Podrazumevani pragovi su:

- `very_similar`: score najmanje 0,94;
- `probable_duplicate`: score najmanje 0,95;
- `very_likely_duplicate`: score najmanje 0,97.

Radi brÅ¾eg izvrÅ¡avanja podrazumevano se Å¡alje 1.000 reproduktivno izabranih upita nad kolekcijom od 113.000 slika. Zbog toga je rezultat analiza uzorka, a ne kompletan spisak svih moguÄ‡ih duplikata. Opcija `--max-images 0` pokreÄ‡e potpunu, ali veoma sporu analizu.

Primer rezultata za prag 0,94 i `top-k=500`:

- 1.657 kandidatskih parova;
- 359 grupa;
- 319 veoma verovatnih duplikata;
- 71 strogi predlog za uklanjanje.

Predlozi nisu automatske odluke. Grupe treba pregledati u HTML izveÅ¡taju pre pokretanja `build-clean-dataset`. Parovi `unlabeled`â€“`labeled` mogu poveÄ‡ati broj prikazanih konflikata labela i ne predstavljaju nuÅ¾no greÅ¡ku anotacije.

```powershell
python .\src\08_dataset_cleaning.py analyze --top-k 500
python .\src\08_dataset_cleaning.py analyze --max-images 5000 --top-k 500
python .\src\08_dataset_cleaning.py inspect-group 1
python .\src\08_dataset_cleaning.py build-clean-dataset
python .\src\08_dataset_cleaning.py verify-cleaned
```

Originalne slike se nikada ne briÅ¡u. OÄiÅ¡Ä‡ena kopija pravi se u `data/cleaned/`.

## HNSW benchmark

`src/09_hnsw_benchmark_v3_m.py` poredi:

- NumPy exact pretragu;
- Qdrant exact pretragu;
- Qdrant HNSW za `hnsw_ef = 16, 64, 128`;
- HNSW indekse za `m = 8, 16, 32`;
- pretragu bez filtera i pretragu sa filterom po labeli.

Skripta menja `m`, Äeka ponovno formiranje indeksa, izvrÅ¡ava merenje i na kraju vraÄ‡a originalnu vrednost. Rezultati se Äuvaju u `reports/hnsw_m_benchmark.csv`.

Na uzorku od 50 upita dobijeno je:

| `m` | `hnsw_ef` | Filter | recall@10 | ProseÄno vreme |
|---:|---:|---|---:|---:|
| 8 | 16 | bez filtera | 0,976 | 83,03 ms |
| 8 | 64 | bez filtera | 0,984 | 82,80 ms |
| 8 | 128 | bez filtera | **0,988** | 82,10 ms |
| 16 | 16 | bez filtera | 0,954 | 81,94 ms |
| 16 | 64 | bez filtera | 0,978 | 82,51 ms |
| 32 | 128 | bez filtera | 0,982 | 82,96 ms |
| 8/16/32 | 16/64/128 | label | **1,000** | pribliÅ¾no 82 ms |

Rezultati pokazuju oÄekivani kompromis izmeÄ‘u pribliÅ¾ne i egzaktne pretrage: veÄ‡i `hnsw_ef` uglavnom poveÄ‡ava recall. Filter po labeli smanjuje skup kandidata i u ovom eksperimentu daje recall 1,0. Razlike u vremenu su male jer u lokalnom Docker/REST okruÅ¾enju znaÄajan deo vremena Äine HTTP i batch troÅ¡kovi. Qdrant exact i NumPy mogu dati malo drugaÄiji redosled suseda kada su cosine score vrednosti veoma bliske.

```powershell
python .\src\09_hnsw_benchmark_v3_m.py
python .\src\09_hnsw_benchmark_v3_m.py --queries 200
```

Promena `m` zahteva ponovno formiranje indeksa, pa benchmark moÅ¾e trajati nekoliko minuta.

## Instalacija i prvo pokretanje

Iz korena repozitorijuma u PowerShell-u:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item infra\.env.example infra\.env
docker compose -f infra\docker-compose.yml up -d
```

Kompletan pipeline pokreÄ‡e se sledeÄ‡im redosledom:

```powershell
python .\scripts\prepare_dataset.py
python .\src\generate_embeddings.py --batch-size 16 --device auto
python .\src\check_embeddings.py
python .\src\02_create_collection.py --recreate
python .\src\04_import_to_qdrant.py
python .\src\05_verify_import.py
```

Generisanje embeddinga je najbrÅ¾e na NVIDIA GPU-u sa CUDA verzijom PyTorch-a. Import koristi `upload_points`, batch veliÄinu 500 i Äetiri paralelna radnika. Ako se import prekine, moÅ¾e se ponovo pokrenuti jer se pointovi upisuju prema stabilnim ID vrednostima.

## UI

UI se pokreÄ‡e komandom:

```powershell
python .\ui\server.py
```

ili:

```powershell
.\START_UI.bat
```

Server je dostupan na `http://127.0.0.1:8765`. Pretraga sliÄnosti podrazumevano koristi HNSW sa `hnsw_ef=64` i uklanja samu query sliku iz rezultata. Exact reÅ¾im ostaje dostupan kroz API parametar `exact=true`.

## Validacija

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src scripts ui tests
python .\src\07_error_analysis.py validate --backend qdrant
python .\src\08_dataset_cleaning.py validate --backend qdrant
```

## Struktura repozitorijuma

```text
data/           metapodaci, lokalne slike i embedding artefakti
infra/          Docker Compose konfiguracija za Qdrant
reports/        CSV, JSON i HTML rezultati analiza
scripts/        priprema dataseta i pomoÄ‡ne skripte
src/            embedding pipeline, Qdrant upiti i analize
tests/          unit i integracioni testovi
ui/             lokalni HTTP server i statiÄki interfejs
```

Veliki generisani fajlovi, slike, embedding matrice, Qdrant podaci, `.env`, `.venv` i HTML izveÅ¡taji ne Äuvaju se na Gitu. Mali benchmark CSV moÅ¾e da se saÄuva kao rezultat eksperimenta.

## OgraniÄenja

- CLIP se koristi bez dodatnog treniranja ili fine-tuninga.
- Unlabeled slike ne mogu da uÄestvuju u raÄunanju klasifikacione taÄnosti.
- Analize sa podrazumevanih 1.000 upita predstavljaju uzorak, ne kompletan prolazak kroz dataset.
- Pragovi za duplikate i dijagnoze greÅ¡aka su heuristike i zahtevaju ruÄni pregled.
- Benchmark sa 50 upita dovoljan je za demonstraciju, ali veÄ‡i uzorak daje stabilnije rezultate.
- Lokalni UI nema autentifikaciju i namenjen je iskljuÄivo lokalnom koriÅ¡Ä‡enju.
