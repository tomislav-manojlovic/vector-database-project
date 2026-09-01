# Qdrant pretraga i analiza STL-10 slika

## Pregled projekta

Projekat povezuje računarski vid i vektorske baze podataka. Za slike iz celog STL-10 dataseta generišu se CLIP embedding vektori, koji se zatim čuvaju i indeksiraju u Qdrantu. Sistem podržava pretragu sličnih slika, analizu klasifikacionih grešaka, pronalaženje duplikata i lokalni veb interfejs.

Implementirana su tri nivoa projekta:

1. pretraga vizuelno sličnih slika;
2. klasifikacija weighted k-NN metodom i analiza grešaka;
3. pronalaženje približnih duplikata i pravljenje očišćene kopije dataseta.

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

Slike iz unlabeled skupa ostaju u bazi jer doprinose realnoj veličini kolekcije i formiranju HNSW indeksa. U metapodacima imaju:

- `label="unlabeled"`;
- `label_id=-1`;
- `is_labeled=False`;
- `split="unlabeled"`.

Skripta `scripts/prepare_dataset.py` automatski preuzima STL-10 pomoću `torchvision`, izvozi slike i pravi:

- `data/metadata.csv` – metapodatke za svih 113.000 slika;
- `data/metadata_sample.csv` – mali uzorak metapodataka;
- `data/images/stl10/` – lokalno izvezene slike koje se ne čuvaju na Gitu.

Dataset nije potrebno ručno preuzimati.

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
     slične slike  analiza grešaka  duplikati
```

## Embedding pipeline

`src/generate_embeddings.py` učitava slike u batch-evima, prosleđuje ih CLIP modelu i čuva rezultate u `data/embeddings/`:

- `embeddings.npy` – matrica oblika približno `113000 x 512`;
- `embeddings_metadata.csv` – ID, putanja, labela, split, `is_labeled`, indeks embeddinga i status;
- `embedding_config.json` – korišćeni model i konfiguracija embeddinga.

Skripta podržava CPU i CUDA. Opcija `--device auto` automatski bira GPU ako CUDA verzija PyTorch-a i odgovarajući NVIDIA drajver postoje.

## Qdrant kolekcija

Docker Compose pokreće `qdrant/qdrant:v1.18.2`. Kolekcija ima sledeću konfiguraciju:

| Podešavanje | Vrednost |
|---|---|
| Kolekcija | `stl10_clip_images` |
| Vektorska dimenzija | 512 |
| Metrika | Cosine |
| Broj pointova | približno 113.000 |
| Payload | `id`, `image_path`, `label`, `label_id`, `is_labeled`, `split` |
| Payload indeksi | `label`, `split`, `is_labeled` |
| Import batch | 500 |
| Paralelni import radnici | 4 |

Velika kolekcija prelazi Qdrantov prag za indeksiranje, pa se umesto običnog full scan-a formira HNSW indeks.

## Exact i HNSW pretraga

Postoje dva režima pretrage:

- **exact** proverava sve odgovarajuće vektore i koristi se kao referentni rezultat;
- **HNSW** koristi približni indeks i podrazumevani je režim u analizama i UI serveru.

Za HNSW pretragu važni su:

- `hnsw_ef` – broj kandidata razmatranih tokom upita; veća vrednost obično povećava recall, ali može povećati vreme pretrage;
- `m` – broj veza čvora u HNSW grafu; veća vrednost pravi gušći indeks, zauzima više memorije i može poboljšati recall;
- `recall@k` – udeo tačnih top-k suseda koje je približna pretraga pronašla u odnosu na exact rezultat.

Skripte `07_error_analysis.py` i `08_dataset_cleaning.py` koriste HNSW sa `hnsw_ef=64`, dok opcija `--exact` uključuje egzaktnu pretragu. Komande `compare-backends` namerno koriste exact režim radi poređenja sa NumPy backendom.

## Nivo 1: pretraga sličnih slika

`src/06_queries.py` podržava dohvatanje pointa, payload filtere, pretragu sličnosti i kontrolisane CRUD operacije.

Kod komande `similar` upitna slika se uklanja iz rezultata, tako da se ne prikazuje kao sopstveni sused sa score vrednošću 1.0. Isto ponašanje koristi UI server.

```powershell
python .\src\06_queries.py similar 1 --top-k 5
python .\src\06_queries.py similar 1 --top-k 5 --label bird
```

## Nivo 2: analiza grešaka

`src/07_error_analysis.py` koristi weighted k-NN nad CLIP vektorima. Sama query slika se isključuje, a cosine score suseda koristi se kao težina glasa klase.

Pošto 100.000 unlabeled slika nema stvarnu klasu, analiza tačnosti radi samo nad 13.000 označenih slika. Radi bržeg izvršavanja podrazumevano se bira reproduktivan uzorak od 1.000 query slika, dok se susedi traže među svih 13.000 označenih slika. Ovo je leave-one-out analiza CLIP prostora, a ne standardna evaluacija zasebno treniranog klasifikatora.

Primer dobijenog rezultata za 1.000 upita:

- 990 tačnih predikcija;
- 10 grešaka;
- tačnost 99,00%;
- dijagnoze: `ambiguous_or_outlier`, `boundary_case` i `class_confusion`.

Izveštaji se čuvaju u `reports/error_analysis/` kao HTML, JSON i CSV fajlovi.

```powershell
python .\src\07_error_analysis.py analyze
python .\src\07_error_analysis.py analyze --max-images 2000
python .\src\07_error_analysis.py analyze --max-images 0
python .\src\07_error_analysis.py analyze --exact
```

Vrednost `--max-images 0` analizira svih 13.000 označenih slika i zato traje znatno duže.

## Nivo 3: pronalaženje duplikata

`src/08_dataset_cleaning.py` traži veoma slične parove i grupiše ih u povezane komponente. Podrazumevani pragovi su:

- `very_similar`: score najmanje 0,94;
- `probable_duplicate`: score najmanje 0,95;
- `very_likely_duplicate`: score najmanje 0,97.

Radi bržeg izvršavanja podrazumevano se šalje 1.000 reproduktivno izabranih upita nad kolekcijom od 113.000 slika. Zbog toga je rezultat analiza uzorka, a ne kompletan spisak svih mogućih duplikata. Opcija `--max-images 0` pokreće potpunu, ali veoma sporu analizu.

Primer rezultata za prag 0,94 i `top-k=500`:

- 1.657 kandidatskih parova;
- 359 grupa;
- 319 veoma verovatnih duplikata;
- 71 strogi predlog za uklanjanje.

Predlozi nisu automatske odluke. Grupe treba pregledati u HTML izveštaju pre pokretanja `build-clean-dataset`. Parovi `unlabeled`–`labeled` mogu povećati broj prikazanih konflikata labela i ne predstavljaju nužno grešku anotacije.

```powershell
python .\src\08_dataset_cleaning.py analyze --top-k 500
python .\src\08_dataset_cleaning.py analyze --max-images 5000 --top-k 500
python .\src\08_dataset_cleaning.py inspect-group 1
python .\src\08_dataset_cleaning.py build-clean-dataset
python .\src\08_dataset_cleaning.py verify-cleaned
```

Originalne slike se nikada ne brišu. Očišćena kopija pravi se u `data/cleaned/`.

## HNSW benchmark

`src/09_hnsw_benchmark_v3_m.py` poredi:

- NumPy exact pretragu;
- Qdrant exact pretragu;
- Qdrant HNSW za `hnsw_ef = 16, 64, 128`;
- HNSW indekse za `m = 8, 16, 32`;
- pretragu bez filtera i pretragu sa filterom po labeli.

Skripta menja `m`, čeka ponovno formiranje indeksa, izvršava merenje i na kraju vraća originalnu vrednost. Rezultati se čuvaju u `reports/hnsw_m_benchmark.csv`.

Na uzorku od 50 upita dobijeno je:

| `m` | `hnsw_ef` | Filter | recall@10 | Prosečno vreme |
|---:|---:|---|---:|---:|
| 8 | 16 | bez filtera | 0,976 | 83,03 ms |
| 8 | 64 | bez filtera | 0,984 | 82,80 ms |
| 8 | 128 | bez filtera | **0,988** | 82,10 ms |
| 16 | 16 | bez filtera | 0,954 | 81,94 ms |
| 16 | 64 | bez filtera | 0,978 | 82,51 ms |
| 32 | 128 | bez filtera | 0,982 | 82,96 ms |
| 8/16/32 | 16/64/128 | label | **1,000** | približno 82 ms |

Rezultati pokazuju očekivani kompromis između približne i egzaktne pretrage: veći `hnsw_ef` uglavnom povećava recall. Filter po labeli smanjuje skup kandidata i u ovom eksperimentu daje recall 1,0. Razlike u vremenu su male jer u lokalnom Docker/REST okruženju značajan deo vremena čine HTTP i batch troškovi. Qdrant exact i NumPy mogu dati malo drugačiji redosled suseda kada su cosine score vrednosti veoma bliske.

```powershell
python .\src\09_hnsw_benchmark_v3_m.py
python .\src\09_hnsw_benchmark_v3_m.py --queries 200
```

Promena `m` zahteva ponovno formiranje indeksa, pa benchmark može trajati nekoliko minuta.

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

Kompletan pipeline pokreće se sledećim redosledom:

```powershell
python .\scripts\prepare_dataset.py
python .\src\generate_embeddings.py --batch-size 16 --device auto
python .\src\check_embeddings.py
python .\src\02_create_collection.py --recreate
python .\src\04_import_to_qdrant.py
python .\src\05_verify_import.py
```

Generisanje embeddinga je najbrže na NVIDIA GPU-u sa CUDA verzijom PyTorch-a. Import koristi `upload_points`, batch veličinu 500 i četiri paralelna radnika. Ako se import prekine, može se ponovo pokrenuti jer se pointovi upisuju prema stabilnim ID vrednostima.

## UI

UI se pokreće komandom:

```powershell
python .\ui\server.py
```

ili:

```powershell
.\START_UI.bat
```

Server je dostupan na `http://127.0.0.1:8765`. Pretraga sličnosti podrazumevano koristi HNSW sa `hnsw_ef=64` i uklanja samu query sliku iz rezultata. Exact režim ostaje dostupan kroz API parametar `exact=true`.

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
scripts/        priprema dataseta i pomoćne skripte
src/            embedding pipeline, Qdrant upiti i analize
tests/          unit i integracioni testovi
ui/             lokalni HTTP server i statički interfejs
```

Veliki generisani fajlovi, slike, embedding matrice, Qdrant podaci, `.env`, `.venv` i HTML izveštaji ne čuvaju se na Gitu. Mali benchmark CSV može da se sačuva kao rezultat eksperimenta.

## Ograničenja

- CLIP se koristi bez dodatnog treniranja ili fine-tuninga.
- Unlabeled slike ne mogu da učestvuju u računanju klasifikacione tačnosti.
- Analize sa podrazumevanih 1.000 upita predstavljaju uzorak, ne kompletan prolazak kroz dataset.
- Pragovi za duplikate i dijagnoze grešaka su heuristike i zahtevaju ručni pregled.
- Benchmark sa 50 upita dovoljan je za demonstraciju, ali veći uzorak daje stabilnije rezultate.
- Lokalni UI nema autentifikaciju i namenjen je isključivo lokalnom korišćenju.
