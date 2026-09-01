# Qdrant Image Lab

Studentski projekat iz baza podataka i računarskog vida koji prikazuje kako se slike mogu predstaviti vektorima, sačuvati u vektorskoj bazi i pretraživati prema vizuelnoj sličnosti.

Projekat koristi:

- **STL-10** kao skup slika;
- **CLIP** za generisanje vektora dimenzije 512;
- **Qdrant** kao vektorsku bazu podataka;
- **weighted k-NN** za klasifikaciju i analizu grešaka;
- jednostavan lokalni **HTML/CSS/JavaScript UI** sa Python HTTP serverom.

## Šta projekat omogućava

Kroz korisnički interfejs mogu da se demonstriraju:

1. pregled stanja dataseta, embeddinga i Qdrant kolekcije;
2. pretraga najsličnijih slika i filtriranje prema klasi;
3. bezbedan CRUD nad privremenim demo pointovima;
4. evaluacija weighted k-NN modela i analiza pogrešnih klasifikacija;
5. pronalaženje veoma sličnih slika i pravljenje očišćene kopije dataseta;
6. testiranje stvarnih podataka iz lokalne Qdrant baze;
7. kratka prezentacija kompletnog toka projekta.

## Tok podataka

```text
STL-10 slike
     |
     v
metadata.csv
     |
     v
CLIP embedding (512 brojeva po slici)
     |
     v
Qdrant kolekcija (Cosine sličnost)
     |
     +--> pretraga sličnih slika
     +--> payload filter i CRUD
     +--> weighted k-NN i greške modela
     +--> slični parovi i čišćenje dataseta
     |
     v
lokalni veb interfejs
```

Jedan Qdrant point predstavlja jednu sliku i sadrži:

- jedinstveni ID;
- CLIP vektor dimenzije 512;
- payload sa putanjom slike, labelom i pomoćnim podacima.

Za poređenje vektora koristi se **Cosine** metrika. Veći score znači da CLIP smatra dve slike vizuelno ili semantički sličnijim.

## Podaci koji se koriste

Kompletna Qdrant kolekcija sadrži **113.000 stvarnih STL-10 slika**:

| Deo skupa | Broj slika | Labele |
|---|---:|---|
| `train` | 5.000 | 10 klasa |
| `test` | 8.000 | 10 klasa |
| `unlabeled` | 100.000 | bez poznate klase |
| **Ukupno** | **113.000** | 10 klasa + `unlabeled` |

Deset labeliranih klasa su: `airplane`, `bird`, `car`, `cat`, `deer`, `dog`, `horse`, `monkey`, `ship` i `truck`.

Interaktivna analiza grešaka i analiza kvaliteta koriste uravnotežen uzorak od **1.000 stvarnih slika**, odnosno po 100 labeliranih slika iz svake klase. Uzorak nije mock i čita se iz istih metapodataka, embeddinga i Qdrant kolekcije kao ostatak aplikacije.

Uzorak od 1.000 slika je izabran zato što:

- svaka klasa ima isti broj primera;
- analiza se dovoljno brzo izvršava tokom demonstracije;
- rezultat je ponovljiv i lak za objašnjavanje;
- poređenje svih mogućih parova nad 113.000 slika bilo bi nepotrebno sporo za studentsku demonstraciju.

Rezultati u UI-ju nisu ručno upisani. Broj grešaka, tačnost, broj sličnih parova, grupa i kandidata čitaju se iz poslednje pokrenute analize.

## Preduslovi

Pre prvog pokretanja potrebno je imati:

- Windows i PowerShell;
- Python 3.10 ili kompatibilnu noviju verziju;
- Docker Desktop;
- internet vezu za prvo preuzimanje STL-10 dataseta, Docker image-a i CLIP modela;
- dovoljno slobodnog prostora za 113.000 slika i njihove embeddinge.

Sve naredne komande pokreću se iz **korena projekta**:

```text
D:\BazeVidProj\vector-database-project
```

Ako je terminal trenutno u `src` direktorijumu, prvo se treba vratiti jedan nivo:

```powershell
cd ..
```

## Prvo pokretanje projekta

Ovaj postupak se radi prilikom prvog podešavanja na novom računaru ili nakon potpuno novog kloniranja projekta.

### 1. Otvoriti Docker Desktop

Sačekati da Docker Desktop završi pokretanje. Qdrant će kasnije biti pokrenut iz deploy skripte.

### 2. Napraviti Python virtuelno okruženje

U PowerShell terminalu, iz korena projekta:

```powershell
py -m venv .venv
```

Aktiviranje okruženja nije obavezno zato što naredne komande direktno koriste Python iz `.venv` direktorijuma.

### 3. Instalirati Python biblioteke

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Napraviti lokalni konfiguracioni fajl

```powershell
Copy-Item infra\.env.example infra\.env
```

Za lokalni rad nije potrebno menjati podrazumevane vrednosti. Qdrant je dostupan samo na lokalnom računaru preko adrese `http://localhost:6333`.

Ako `infra\.env` već postoji, ovaj korak se preskače.

### 5. Preuzeti i pripremiti STL-10 dataset

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
```

Skripta preuzima STL-10, izvozi slike i pravi `data\metadata.csv`.

### 6. Generisati CLIP embeddinge

```powershell
.\.venv\Scripts\python.exe src\generate_embeddings.py
```

Ovo je najduži korak jer se obrađuje 113.000 slika. Rezultati se čuvaju u `data\embeddings\` i ne generišu se ponovo pri svakom pokretanju.

Ako su `data\metadata.csv`, `data\embeddings\embeddings.npy` i ostali embedding fajlovi već pripremljeni i odgovaraju trenutnom datasetu, koraci 5 i 6 mogu da se preskoče.

### 7. Pokrenuti linearni deploy

```powershell
powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1
```

Deploy skripta izvršava korake redom i zaustavlja se ako neki od njih ne uspe:

1. pokreće Qdrant kontejner;
2. čeka da Qdrant bude spreman;
3. proverava lokalne embedding fajlove;
4. kreira kolekciju ako ne postoji;
5. uvozi podatke u Qdrant;
6. proverava broj pointova i konfiguraciju kolekcije;
7. pokreće korisnički interfejs.

Na kraju se u pregledaču otvara:

```text
http://127.0.0.1:8765
```

Ako je port `8765` zauzet, server bira prvi slobodan port iz narednih 30 portova i ispisuje njegovu adresu u terminalu.

## Svako naredno pokretanje

Kada su dataset, embedding fajlovi i Qdrant kolekcija već napravljeni, nije potrebno ponovo generisati podatke niti raditi kompletan deploy.

### Preporučeni postupak

1. Pokrenuti Docker Desktop.
2. Otvoriti PowerShell u korenu projekta.
3. Pokrenuti postojeći Qdrant kontejner:

```powershell
docker compose -f infra\docker-compose.yml up -d
```

4. Pokrenuti UI:

```powershell
.\START_UI.bat
```

Pošto Qdrant kontejner ima pravilo `restart: unless-stopped`, često je dovoljno pokrenuti Docker Desktop i zatim `START_UI.bat`. Posebna Docker komanda iznad je bezbedna provera da je kontejner zaista pokrenut.

### Kada ponovo koristiti deploy skriptu

`scripts\deploy.ps1` treba ponovo pokrenuti kada:

- Qdrant kolekcija još ne postoji;
- obrisan je Docker volume;
- promenjeni su metadata ili embedding fajlovi;
- potrebno je ponovo uvesti i proveriti sve pointove;
- projekat se podešava na novom računaru.

Za obično otvaranje aplikacije deploy nije potreban.

## Zaustavljanje aplikacije

UI se zaustavlja u terminalu kombinacijom:

```text
Ctrl+C
```

Ako se pojavi pitanje:

```text
Terminate batch job (Y/N)?
```

unosi se `Y` i pritisne Enter. To samo zaustavlja UI proces i ne briše podatke.

Qdrant se opciono može zaustaviti komandom:

```powershell
docker compose -f infra\docker-compose.yml stop
```

Podaci ostaju sačuvani u Docker volumenu i dostupni su pri sledećem pokretanju.

## Delovi korisničkog interfejsa

### Pregled

Prikazuje broj slika, dimenziju embeddinga, stanje Qdranta, poslednje rezultate analiza i testove stvarnih podataka.

### Pretraga

Omogućava dohvatanje slike prema ID-u, payload filtriranje prema labeli i prikaz najsličnijih slika.

### CRUD demo

Create, Read, Update i Delete izvršavaju se samo nad privremenim pointovima čiji ID počinje od `9.000.000`. Originalni dataset pointovi su zaštićeni od izmene i brisanja.

### Evaluacija modela

Weighted k-NN za svaku od 1.000 analiziranih slika pronalazi pet najbližih drugih labeliranih slika. Kosinusni score suseda koristi se kao težina glasa klase.

Stranica prikazuje:

- broj tačnih i pogrešnih klasifikacija;
- ukupnu tačnost;
- matricu konfuzije;
- susede koji objašnjavaju svaku pogrešnu odluku.

CLIP nije dodatno treniran u ovom projektu. Klasifikacija se obavlja weighted k-NN metodom direktno nad postojećim CLIP vektorima.

### Kvalitet dataseta

Qdrant traži jedinstvene parove slika čija je kosinusna sličnost najmanje `0.94`.

Podrazumevani pragovi su:

| Kategorija | Prag |
|---|---:|
| sličan par | `0.94` |
| verovatan duplikat | `0.95` |
| veoma verovatan duplikat | `0.97` |

Ista slika može da učestvuje u više parova, pa broj parova nije isto što i broj slika. Povezani parovi spajaju se u grupe. Jedna slika iz grupe bira se kao reprezentativna, a ostale dobijaju preporuku:

- `keep` – zadržati reprezentativnu sliku;
- `review` – ručno pregledati;
- `remove_candidate` – strogi kandidat za izostavljanje iz očišćene kopije.

Originalni dataset se nikada automatski ne briše. Očišćeni podaci se prave kao posebna kopija u `data\cleaned\`.

### Prezentacija

„Brzi demo” redom prikazuje stanje dataseta, Qdrant kolekciju, similarity search, payload filter, poslednju analizu grešaka i poslednju analizu kvaliteta. Prikazani brojevi dolaze iz trenutnih podataka i izveštaja.

## Pokretanje analiza iz terminala

Iste analize koje pokreće UI mogu ručno da se pokrenu iz korena projekta.

### Analiza grešaka modela

```powershell
.\.venv\Scripts\python.exe src\07_error_analysis.py analyze --backend qdrant --k 5
```

Izveštaji se čuvaju u:

```text
reports\error_analysis\
```

### Analiza kvaliteta dataseta

```powershell
.\.venv\Scripts\python.exe src\08_dataset_cleaning.py analyze --backend qdrant
```

Izveštaji se čuvaju u:

```text
reports\variant3_dataset_cleaning\
```

### Pravljenje i provera očišćene kopije

```powershell
.\.venv\Scripts\python.exe src\08_dataset_cleaning.py build-clean-dataset
.\.venv\Scripts\python.exe src\08_dataset_cleaning.py verify-cleaned
```

## Testiranje

Dugme **Pokreni testove** na glavnoj stranici pokreće testove nad stvarnim lokalnim podacima i stvarnom Qdrant kolekcijom. Ti testovi proveravaju:

- broj pointova i dimenziju kolekcije;
- slaganje stvarnog vektora sa lokalnim embedding fajlom;
- payload filter i similarity search;
- weighted k-NN nad stvarnim susedima.

Isti testovi mogu da se pokrenu iz terminala:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_real_database -v
```

Svi testovi zajedno (unit testovi i testovi stvarne baze) pokreću se komandom:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Struktura direktorijuma

```text
vector-database-project/
|
|-- data/
|   |-- images/stl10/                 izvezene STL-10 slike
|   |-- embeddings/                   CLIP vektori i njihovi metapodaci
|   |-- cleaned/                      opciona očišćena kopija
|   |-- metadata.csv                  metapodaci svih slika
|   `-- metadata_sample.csv           mali primer metapodataka
|
|-- infra/
|   |-- docker-compose.yml            Qdrant Docker servis
|   |-- .env.example                  primer lokalne konfiguracije
|   `-- scripts/
|       |-- wait_for_qdrant.py         čekanje da baza bude spremna
|       |-- qdrant_connection.py       provera osnovne veze
|       `-- check_qdrant_connection.py dodatna provera konekcije
|
|-- reports/
|   |-- error_analysis/               HTML, CSV i JSON analiza grešaka
|   `-- variant3_dataset_cleaning/     rezultati analize kvaliteta
|
|-- scripts/
|   |-- deploy.ps1                    linearni deploy kompletnog sistema
|   |-- prepare_dataset.py             preuzimanje i priprema STL-10
|   |-- demo_error_analysis.ps1        terminalski demo grešaka
|   `-- demo_variant3.ps1              terminalski demo čišćenja
|
|-- src/
|   |-- 02_create_collection.py       kreiranje Qdrant kolekcije
|   |-- 04_import_to_qdrant.py        paketni import pointova
|   |-- 05_verify_import.py           provera uspešnog importa
|   |-- 06_queries.py                 search, filter i CRUD komande
|   |-- 07_error_analysis.py          weighted k-NN i analiza grešaka
|   |-- 08_dataset_cleaning.py        slični parovi i očišćena kopija
|   |-- 09_hnsw_benchmark.py          opciono poređenje HNSW pretrage
|   |-- check_embeddings.py           provera embedding fajlova
|   |-- generate_embeddings.py        generisanje CLIP vektora
|   |-- model_utils.py                pomoćne funkcije za CLIP
|   `-- qdrant_common.py              zajednička Qdrant podešavanja
|
|-- tests/
|   |-- test_real_database.py         testovi stvarnih podataka i Qdranta
|   |-- test_queries.py               testovi upita
|   |-- test_error_analysis.py        testovi analize grešaka
|   |-- test_dataset_cleaning.py      testovi analize kvaliteta
|   `-- test_ui_server.py             testovi serverske logike
|
|-- ui/
|   |-- index.html                    struktura korisničkog interfejsa
|   |-- styles.css                    izgled aplikacije
|   |-- app.js                        UI logika i API pozivi
|   `-- server.py                     lokalni HTTP server i API
|
|-- START_UI.bat                      jednostavno kasnije pokretanje UI-ja
|-- requirements.txt                  Python biblioteke
`-- README.md                         dokumentacija projekta
```

Veliki generisani fajlovi, kao što su slike, embedding matrica, izveštaji, `.venv` i Docker podaci, ne čuvaju se u Git repozitorijumu.

## Česti problemi

### `scripts\deploy.ps1` ne postoji

Komanda je verovatno pokrenuta iz `src` direktorijuma. Vratiti se u koren projekta:

```powershell
cd ..
powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1
```

### Qdrant nije povezan

Proveriti da li Docker Desktop radi, a zatim pokrenuti:

```powershell
docker compose -f infra\docker-compose.yml up -d
```

### Stranice analiza su prazne

Izveštaji još nisu generisani. Na odgovarajućoj stranici kliknuti **Pokreni analizu** ili **Ponovi analizu**.

### UI javlja da komanda traje predugo

Proveriti da li Qdrant radi i da li lokalni embedding fajlovi odgovaraju kolekciji. Analize su ograničene na 1.000 slika kako bi mogle da se završe u vremenu pogodnom za demonstraciju.

## Ograničenja projekta

- CLIP nije dodatno treniran za STL-10.
- Weighted k-NN je jednostavna i objašnjiva metoda, a ne posebno treniran klasifikator.
- Pragovi za slične slike predstavljaju heuristike i zahtevaju vizuelni pregled.
- Rezultati analiza važe za izabrani uzorak od 1.000 labeliranih slika.
- Lokalni UI nema autentifikaciju i namenjen je samo radu na `127.0.0.1`.
- Ispravan rad zahteva da metadata, embedding fajlovi i Qdrant kolekcija predstavljaju isti skup podataka.
