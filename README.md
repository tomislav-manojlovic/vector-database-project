# Qdrant alat za računarski vid

## Pregled projekta

Ovaj projekat objedinjuje računarski vid i vektorsku bazu podataka u jedan reproduktivan sistem za analizu slika. Sistem priprema uravnotežen podskup STL-10 trening skupa, svaku sliku predstavlja normalizovanim CLIP embeddingom, čuva vektore u Qdrantu i omogućava pretragu, analizu grešaka modela, čišćenje skupa podataka i kontrolisane CRUD operacije putem komandne linije i lokalnog veb interfejsa.

Implementirana su sva tri zahtevana nivoa projekta:

1. **Pretraga sličnih slika** koristi kosinusnu sličnost u Qdrantu za pronalaženje slika bliskih upitnom vektoru.
2. **Analiza grešaka modela** koristi najbliže susede za klasifikaciju slika i objašnjavanje pogrešnih odluka.
3. **Čišćenje skupa podataka** otkriva grupe veoma sličnih slika i pravi pregledanu, odvojenu očišćenu kopiju bez menjanja izvornog skupa.

Embedding slike je sažet numerički prikaz njenog vizuelnog sadržaja. Slike koje CLIP smatra vizuelno ili semantički povezanim imaju bliske vektore dimenzije 512. Zbog toga isti prikaz može da se koristi za pretragu, lokalnu analizu klasifikacije i otkrivanje redundantnih podataka.

## Glavne funkcionalnosti

- reproduktivno preuzimanje STL-10 skupa, izvoz slika i generisanje metapodataka;
- paketno generisanje CLIP embeddinga uz obradu neispravnih slika;
- kreiranje Qdrant kolekcije, indeksiranje payload-a, paketni import i verifikacija;
- dohvatanje pointova, filtriranje payload-a, kosinusna pretraga i kontrolisane CRUD operacije;
- klasifikacija ponderisanim k najbližih suseda i dijagnostika grešaka;
- precizni lokalni NumPy backend za offline analizu i poređenje sa Qdrantom;
- grupisanje duplikata i približnih duplikata pomoću podesivih pragova;
- evidentiranje odluka pregleda i pravljenje odvojenog očišćenog skupa;
- veb interfejs bez dodatnog framework-a, poslužen Python standardnom bibliotekom.

## Arhitektura sistema

```text
STL-10 trening skup
        |
        v
JPEG slike + data/metadata.csv
        |
        v
CLIP openai/clip-vit-base-patch32
        |
        v
L2-normalizovani vektori dimenzije 512
        |
        +---------------------> NumPy offline validacija
        |
        v
Qdrant: stl10_clip_images (Cosine)
        |
        +--> pretraga sličnosti i CRUD
        +--> weighted k-NN analiza grešaka
        +--> grupisanje sličnih slika i čišćenje
        |
        v
Python HTTP server <--> statički HTML/CSS/JavaScript interfejs
```

UI server u `ui/server.py` poslužuje statičke fajlove, direktno poziva Qdrant za interaktivne upite, čita generisane CSV/JSON izveštaje i pokreće skripte za analizu kao podprocese. Server se vezuje za `127.0.0.1`. Podrazumevani traženi port je `8765`, a ako je zauzet, bira se prvi slobodan port u narednih 30 portova.

## Skup podataka

Skripta `scripts/prepare_dataset.py` preuzima **STL-10 trening skup** pomoću `torchvision.datasets.STL10`. Bira prvih 100 primera iz svake od deset klasa — airplane, bird, car, cat, deer, dog, horse, monkey, ship i truck — odnosno ukupno 1.000 slika.

Skripta ponovo pravi `data/images/stl10/`, upisuje RGB JPEG slike grupisane po klasama i generiše:

- `data/metadata.csv`: ID, relativnu putanju slike, labelu, numeričku labelu, naziv skupa, podelu, izvorni indeks i naziv fajla;
- `data/metadata_sample.csv`: po tri primera metapodataka za svaku klasu.

Pokretanje skripte za pripremu zamenjuje postojeći direktorijum izvezenih slika. Preuzeti izvorni skup i izvezene slike predstavljaju generisane resurse i ignorisani su u Gitu.

## Pipeline za generisanje embeddinga

`src/generate_embeddings.py` učitava `openai/clip-vit-base-patch32` pomoću biblioteke Hugging Face Transformers. Skripta razrešava putanju svakog metapodatka, obrađuje ispravne slike u paketima, dobija CLIP obeležja, pretvara ih u `float32` i L2-normalizuje svaki vektor. Nedostajuće ili nečitljive slike evidentiraju se kao preskočene i ne dodeljuje im se vektor.

Podrazumevani izlazni direktorijum `data/embeddings/` sadrži:

- `embeddings.npy`: matricu vektora oblika `N x 512`;
- `embeddings_metadata.csv`: polja `id`, `image_path`, `label`, `embedding_index` i `status`;
- `embedding_config.json`: naziv modela, dimenziju, informaciju o normalizaciji, metriku, broj slika i vreme generisanja.

U Qdrant se uvoze samo redovi metapodataka čiji je status `ok`. Dostavljena konfiguracija opisuje 1.000 normalizovanih vektora dimenzije 512.

## Vektorska baza

Docker Compose pokreće `qdrant/qdrant:v1.18.2`. HTTP i gRPC su podrazumevano lokalno dostupni na portovima `6333` i `6334`, dok se trajni podaci baze čuvaju u imenovanom Docker volumenu `qdrant-image-search-storage`.

Aplikacija koristi sledeći ugovor kolekcije:

| Podešavanje | Vrednost |
|---|---|
| Podrazumevana kolekcija | `stl10_clip_images` |
| Dimenzija vektora | `512` |
| Metrika rastojanja | Cosine |
| Payload | `id`, `image_path`, `label` |
| Payload indeks | keyword indeks nad poljem `label` |
| Veličina import paketa | `100` |

Vrednosti `QDRANT_URL`, `QDRANT_API_KEY` i `QDRANT_COLLECTION` mogu da se zadaju u korenom `.env` fajlu ili u `infra/.env`. Dimenzija vektora i metrika fiksirane su Python ugovorom kolekcije. `infra/.env.example` dokumentuje i promenljive za Docker adresu i portove.

## Implementirani nivoi projekta

### Nivo 1: pretraga sličnih slika

`src/06_queries.py` može da dohvati sačuvani vektor prema ID-u pointa i pošalje ga Qdrantu kao kosinusni upit. Sama upitna slika isključuje se iz rezultata, rezultati su poređani prema score vrednosti, a opcioni `label` filter ograničava prostor pretrage. Isti modul može CLIP modelom da generiše embedding nove slike, upiše je kao Qdrant point i zatim izvrši pretragu.

Sloj za upite podržava i dohvatanje pointa, filtriranje po labeli, izmenu payload-a, eksplicitno brisanje uz potvrdu i bezbednu demonstraciju brisanja privremenog pointa.

### Nivo 2: analiza grešaka modela

Projekat ne sadrži checkpoint posebno treniranog klasifikatora. `src/07_error_analysis.py` implementira ponderisani k-NN klasifikator nad postojećim CLIP prostorom:

1. podrazumevano pronalazi pet najbližih drugih slika;
2. negativne kosinusne score vrednosti ograničava na nulu i koristi ih kao težine glasova klasa;
3. bira klasu sa najvećim ukupnim glasom;
4. poredi predikciju sa STL-10 labelom;
5. za svaku grešku čuva susede i heurističku dijagnozu.

Qdrant upiti se podrazumevano šalju paketno i koriste preciznu pretragu. Lokalni NumPy backend sa kosinusnom sličnošću omogućava offline validaciju i poređenje backend-a.

Dijagnoze grešaka predstavljaju heuristike za ručni pregled, a ne automatske ispravke labela:

- `possible_annotation_issue`: svi susedi podržavaju jednu drugu klasu, nijedan ne podržava stvarnu klasu, a prosečna sličnost je najmanje `0.92`;
- `ambiguous_or_outlier`: najbolji sused ima score ispod `0.55` ili su dokazi suseda na drugi način rasuti;
- `class_confusion`: najmanje 60% suseda podržava predviđenu klasu;
- `boundary_case`: i stvarna i predviđena klasa imaju lokalnu podršku, ali nisu ispunjena stroža pravila.

Analiza upisuje HTML, JSON i CSV artefakte u `reports/error_analysis/`, uključujući predikcije, greške, detalje suseda, matricu konfuzije i metrike po klasama.

### Nivo 3: čišćenje skupa podataka

`src/08_dataset_cleaning.py` traži parove slika slične po kosinusnoj metrici koristeći Qdrant ili lokalni precizni backend. Podrazumevane kategorije su:

- `very_similar`: score najmanje `0.94`;
- `probable_duplicate`: score najmanje `0.95`;
- `very_likely_duplicate`: score najmanje `0.97`.

Kandidatski parovi spajaju se u povezane grupe. Za svaku grupu bira se reprezentativna slika. Slike sa istom labelom koje zadovoljavaju strogi prag mogu biti preporučene kao kandidati za uklanjanje. Parovi sa različitim labelama čuvaju se za ručni pregled, jer sama sličnost ne može da dokaže grešku u anotaciji.

Analiza upisuje tabele za pregled i HTML izveštaj u `reports/variant3_dataset_cleaning/`. Komanda `build-clean-dataset` čita odluke pregleda i pravi odvojenu kopiju u `data/cleaned/`, koja sadrži filtrirane vektore, ponovo mapirane indekse embeddinga, metapodatke, konfiguraciju i manifest čišćenja. Izvorne slike i izvorni embedding fajlovi nikada se ne menjaju.

Ovaj nivo je usmeren na duplikate i veoma redundantne uzorke. Projekat **nema** zaseban globalni postupak za otkrivanje outlier-a u celom skupu; sumnjivi slučajevi sa niskom sličnošću izdvajaju se heuristikom analize grešaka iz nivoa 2.

## Preduslovi

- Windows za priloženi `.bat` launcher; Python komande rade i u drugim podržanim okruženjima;
- Python 3.10 ili druga verzija kompatibilna sa navedenim bibliotekama;
- Docker Desktop sa Docker Compose podrškom za Qdrant;
- dovoljno prostora za STL-10, izvezene slike, keš modela, embeddinge i izveštaje;
- pristup internetu prilikom prvog preuzimanja skupa i CLIP modela.

Python zavisnosti navedene su u `requirements.txt`: pandas, NumPy, PyTorch, torchvision, Transformers, Pillow, tqdm, qdrant-client 1.18.x i python-dotenv.

## Prvo podešavanje

Sledeće komande treba pokrenuti iz korena repozitorijuma u PowerShell-u:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item infra\.env.example infra\.env
docker compose -f infra\docker-compose.yml up -d
```

Kopiranje environment fajla nije obavezno ako podrazumevane vrednosti odgovaraju okruženju. Popunjen `.env` sa pristupnim podacima ne treba commit-ovati.

### Priprema podataka i inicijalizacija Qdranta

Za potpuno ponovno generisanje, faze treba pokrenuti sledećim redosledom:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
.\.venv\Scripts\python.exe src\generate_embeddings.py
.\.venv\Scripts\python.exe src\check_embeddings.py
.\.venv\Scripts\python.exe src\02_create_collection.py --recreate
.\.venv\Scripts\python.exe src\04_import_to_qdrant.py
.\.venv\Scripts\python.exe src\05_verify_import.py
```

Opcija `--recreate` briše i ponovo pravi samo konfigurisanu projektnu kolekciju. Treba je izostaviti kada postojeća kolekcija mora da se sačuva. Priprema skupa i generisanje embeddinga mogu da se preskoče tokom normalnog korišćenja ako generisani fajlovi već postoje i odgovaraju sadržaju kolekcije.

Ne postoji faza treniranja ili fine-tuninga: CLIP je unapred treniran, a klasifikacija koristi ponderisani k-NN u vreme upita.

## Pokretanje kompletne aplikacije

1. Pokrenuti Docker Desktop i proveriti da Qdrant radi:

   ```powershell
   docker compose -f infra\docker-compose.yml up -d
   ```

2. Iz korena repozitorijuma pokrenuti aplikaciju:

   ```powershell
   .\START_UI.bat
   ```

Launcher proverava postojanje `.venv\Scripts\python.exe` i pokreće `ui\server.py`. On **ne pokreće** Docker i ne priprema niti uvozi podatke. Server otvara lokalni dashboard u podrazumevanom pregledaču. U dashboard-u je moguće pregledati stanje sistema, pretraživati prema ID-u ili labeli, izvršavati pretragu sličnosti, koristiti CRUD operacije ograničene na demonstracione ID-eve, ponavljati analize, pregledati greške klasifikacije i grupe suseda, napraviti očišćenu kopiju i pokrenuti testove.

## Pokretanje pojedinačnih tokova

### Pretraga i CRUD

```powershell
.\.venv\Scripts\python.exe src\06_queries.py check
.\.venv\Scripts\python.exe src\06_queries.py get 1 --with-vector
.\.venv\Scripts\python.exe src\06_queries.py filter dog --limit 5
.\.venv\Scripts\python.exe src\06_queries.py similar 1 --top-k 5
.\.venv\Scripts\python.exe src\06_queries.py similar 1 --top-k 5 --label bird
.\.venv\Scripts\python.exe src\06_queries.py create-from-image data\images\stl10\dog\dog_0001.jpg dog --id 2001
.\.venv\Scripts\python.exe src\06_queries.py update 2001 reviewed true
.\.venv\Scripts\python.exe src\06_queries.py delete 2001
.\.venv\Scripts\python.exe src\06_queries.py delete-test
```

### Analiza grešaka

```powershell
.\.venv\Scripts\python.exe src\07_error_analysis.py validate --backend qdrant
.\.venv\Scripts\python.exe src\07_error_analysis.py analyze --backend qdrant --k 5
.\.venv\Scripts\python.exe src\07_error_analysis.py inspect 3 --backend qdrant --k 5
.\.venv\Scripts\python.exe src\07_error_analysis.py compare-backends --sample-size 30 --k 5
powershell -ExecutionPolicy Bypass -File scripts\demo_error_analysis.ps1
```

### Čišćenje skupa podataka

```powershell
.\.venv\Scripts\python.exe src\08_dataset_cleaning.py validate --backend qdrant
.\.venv\Scripts\python.exe src\08_dataset_cleaning.py analyze --backend qdrant
.\.venv\Scripts\python.exe src\08_dataset_cleaning.py inspect-group 1
.\.venv\Scripts\python.exe src\08_dataset_cleaning.py compare-backends
.\.venv\Scripts\python.exe src\08_dataset_cleaning.py build-clean-dataset
.\.venv\Scripts\python.exe src\08_dataset_cleaning.py verify-cleaned
powershell -ExecutionPolicy Bypass -File scripts\demo_variant3.ps1
```

Opcija `--help` svake komande prikazuje dostupne putanje, veličine paketa, backend-e i pragove.

## Struktura repozitorijuma

```text
.
|-- data/                 metapodaci i generisani skupovi/embeddinzi
|-- infra/                Qdrant Compose konfiguracija i alati za povezivanje
|-- reports/              generisani rezultati analiza
|-- scripts/              priprema skupa i demonstracioni tokovi
|-- src/                  embedding, Qdrant, upiti, analiza i čišćenje
|-- tests/                unit i integracioni testovi sa lokalnim podacima
|-- ui/                   lokalni HTTP server i statička veb aplikacija
|-- requirements.txt      manifest Python zavisnosti
`-- START_UI.bat          Windows launcher za UI
```

## Generisani i ignorisani fajlovi

Sledeći resursi namerno su isključeni iz kontrole verzija zato što su preuzeti, lokalni, veliki ili reproduktivni:

- `.venv/`, Python cache, test cache, coverage fajlovi, IDE podešavanja i logovi;
- `.env` i `infra/.env`;
- `data/raw/` i `data/images/`;
- `data/embeddings/*.npy` i privremeni direktorijumi za testiranje embeddinga;
- ceo `data/cleaned/`;
- generisani direktorijumi izveštaja analize grešaka i čišćenja;
- lokalne frontend zavisnosti i build direktorijumi ako se kasnije uvedu.

Izvorni kod, manifesti zavisnosti, `infra/.env.example`, Docker konfiguracija, uzorci metapodataka, konfiguracija i metapodaci embeddinga potrebni za opis importa, kao i startup skripte, treba da ostanu pod kontrolom verzija. Qdrant podatke čuva u imenovanom Docker volumenu, a ne u repozitorijumu.

## Validacija i primer toka

Lake provere koje ne generišu ponovo skup ili model su:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src scripts infra\scripts ui tests
.\.venv\Scripts\python.exe src\check_embeddings.py
.\.venv\Scripts\python.exe src\07_error_analysis.py validate --backend local
.\.venv\Scripts\python.exe src\08_dataset_cleaning.py validate --backend local
```

Za demonstraciju kompletnog toka treba proveriti import, otvoriti UI, izvršiti pretragu sličnosti prema ID-u, pregledati grešku i njene susede, pregledati grupu za čišćenje, napraviti očišćenu kopiju i verifikovati je. Generisani brojevi i rezultati analiza zavise od trenutnih podataka, pragova, odluka pregleda i sadržaja kolekcije, pa dokumentacija ne garantuje konkretne metrike.

## Ograničenja

- Pripremljeni podskup sadrži samo 1.000 slika iz STL-10 trening skupa.
- CLIP se koristi bez fine-tuninga specifičnog za ovaj projekat.
- Ponderisani k-NN je objašnjiva početna metoda, a ne protokol evaluacije treniranog klasifikatora.
- Pragovi sličnosti i kategorije grešaka su heuristike koje zahtevaju vizuelni pregled.
- Čišćenje kopira metapodatke i vektore; izvorni JPEG fajlovi ostaju u prvobitnom generisanom direktorijumu.
- Lokalni UI nema autentifikaciju i namenjen je isključivo korišćenju preko loopback adrese.
- Potpuna funkcionalnost zavisi od međusobno usklađenih lokalnih embeddinga, metapodataka, slika i sadržaja Qdrant kolekcije.

## Moguća unapređenja

- evaluacija pragova na izdvojenom, ručno pregledanom skupu;
- poseban globalni postupak za rangiranje outlier-a;
- podrška za slanje upitne slike kroz veb interfejs;
- reproduktivno treniranje klasifikatora i evaluacija na izdvojenom skupu;
- zajedničko verzionisanje vektorskih kolekcija i embedding artefakata;
- automatizovani end-to-end testovi nad privremenim Qdrant kontejnerom.
