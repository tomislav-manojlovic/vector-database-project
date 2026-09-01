# Seminarski rad: Qdrant vektorska baza za pretragu i analizu slika

**Predmet:** Baze podataka 2  
**Projekat:** Qdrant Image Lab  
**Sistem:** Qdrant  
**Autori:** Andrija Milovanović, Petar Nešić, Tomislav Manojlović, Mihailo Golubović, Igor Perović, Mihailo Mitrović
**Akademska godina:** 2025/2026.

---

## 1. Uvod

Klasične baze podataka najčešće pronalaze podatke prema tačnoj vrednosti, identifikatoru ili uslovu nad kolonama. Kod slika je često potrebno drugačije pitanje: **koje slike su vizuelno najsličnije zadatoj slici?** Za takvu pretragu slika se prvo pretvara u numerički vektor, odnosno embedding, a zatim se u vektorskoj bazi traže najbliži vektori.

U ovom projektu koristi se **Qdrant**, vektorska baza podataka namenjena čuvanju, pretraživanju i upravljanju vektorima sa dodatnim metadata podacima. Qdrant je napisan u programskom jeziku Rust, dostupan je pod Apache 2.0 licencom i podržava HTTP, gRPC i zvanični Python klijent.

Cilj projekta je da se na stvarnom skupu slika prikažu:

- generisanje CLIP embeddinga;
- čuvanje vektora i payload podataka u Qdrantu;
- pretraga najsličnijih slika i filtriranje;
- bezbedne CRUD operacije;
- klasifikacija pomoću weighted k-NN metode;
- analiza pogrešnih klasifikacija;
- pronalaženje veoma sličnih slika i pravljenje očišćene kopije dataseta.

| Osobina | Vrednost u projektu |
|---|---|
| Baza podataka | Qdrant |
| Tip baze | vektorska baza podataka |
| Qdrant verzija | `1.18.2` |
| Klijent | Python `qdrant-client` |
| Dataset | STL-10 |
| Embedding model | `openai/clip-vit-base-patch32` |
| Dimenzija vektora | 512 |
| Metrika | Cosine |
| Broj pointova | 113.000 |
| Deployment | lokalni Docker kontejner |

## 2. Osnovne karakteristike Qdranta

### 2.1. Namena sistema

Qdrant je specijalizovan za similarity search, odnosno pronalaženje vektora koji su najbliži upitnom vektoru. Tipične primene su pretraga slika, semantička pretraga teksta, preporuke i pronalaženje sličnih proizvoda.

Za razliku od relacione baze, u kojoj bi se vektor čuvao kao niz brojeva i ručno upoređivao sa velikim brojem redova, Qdrant ima ugrađene strukture i API namenjene pretrazi najbližih suseda.

### 2.2. Model podataka

Podaci su organizovani u **kolekcije**. Kolekcija je grupa pointova sa istom konfiguracijom vektora i metrike. Osnovna jedinica podataka je **point**, koji sadrži:

1. ID;
2. jedan ili više vektora;
3. opcioni payload u JSON obliku.

U projektu jedan point predstavlja jednu STL-10 sliku:

```text
Point
|-- id: 639
|-- vector: [v1, v2, ..., v512]
`-- payload:
    |-- image_path
    |-- label
    |-- label_id
    |-- is_labeled
    `-- split
```

Vektor opisuje sadržaj slike, dok payload čuva podatke potrebne aplikaciji i filtriranju.

### 2.3. Sličnost i indeksiranje

CLIP vektori u projektu su L2-normalizovani, a Qdrant kolekcija koristi **Cosine** metriku. Veći score znači veću sličnost između upitne slike i rezultata. Score ne predstavlja procenat verovatnoće, već meru bliskosti dva vektora.

Za ubrzanje približne pretrage Qdrant koristi **HNSW** indeks. HNSW organizuje vektore u višeslojni graf kroz koji se brzo dolazi do dobrih kandidata, bez poređenja upita sa svakim vektorom. Važna su dva parametra:

- `m` određuje približan broj veza po čvoru; veća vrednost obično povećava tačnost, ali zahteva više memorije i dužu izgradnju indeksa;
- `hnsw_ef` određuje širinu pretrage; veća vrednost obično daje bolji recall, ali povećava vreme upita.

Projekat sadrži poseban benchmark koji poredi NumPy exact pretragu, Qdrant exact pretragu i HNSW za različite vrednosti `m` i `hnsw_ef`. Posmatraju se `recall@k` i vreme izvršavanja. Benchmark je izdvojen od glavne demonstracije jer promena parametra `m` zahteva ponovno građenje indeksa.

Pored vektorskog indeksa napravljeni su payload indeksi:

| Polje | Tip indeksa | Namena |
|---|---|---|
| `label` | keyword | filtriranje prema klasi slike |
| `split` | keyword | filtriranje prema delu STL-10 skupa |
| `is_labeled` | bool | odvajanje označenih i neoznačenih slika |

### 2.4. Deployment i skladištenje

Qdrant se lokalno pokreće u Docker kontejneru. REST API je dostupan na portu `6333`, a gRPC na portu `6334`. Podaci se čuvaju u imenovanom Docker volumenu, pa gašenje kontejnera ne briše kolekciju.

Projekat koristi jedan lokalni Qdrant čvor zato što je cilj demonstracija principa vektorske baze, a ne produkcioni distribuirani sistem. Servis je vezan za `127.0.0.1` i nije namenjen javnom pristupu.

### 2.5. CRUD operacije

Python klijent omogućava osnovne operacije nad pointovima:

- **Create** – upsert pointa sa ID-em, vektorom i payloadom;
- **Read** – retrieve prema ID-u ili scroll uz payload filter;
- **Update** – izmena odabranih payload polja;
- **Delete** – brisanje pointa prema ID-u.

UI dozvoljava izmene samo nad privremenim demo pointovima čiji je ID najmanje `9.000.000`. Time su originalni STL-10 pointovi zaštićeni od slučajne izmene ili brisanja.

## 3. Implementacija projekta

### 3.1. Dataset i priprema podataka

Korišćen je kompletan STL-10 dataset:

| Split | Broj slika | Napomena |
|---|---:|---|
| `train` | 5.000 | označene slike |
| `test` | 8.000 | označene slike |
| `unlabeled` | 100.000 | slike bez poznate klase |
| **Ukupno** | **113.000** | 10 klasa + neoznačeni deo |

Deset poznatih klasa su: airplane, bird, car, cat, deer, dog, horse, monkey, ship i truck.

Skripta `scripts/prepare_dataset.py` preuzima podatke, izvozi slike i pravi metadata fajl. Zatim `src/generate_embeddings.py` koristi unapred trenirani CLIP model i za svaku ispravnu sliku generiše normalizovan vektor dimenzije 512.

Ne postoji faza treniranja modela. CLIP se koristi kao feature extractor, dok se klasifikacija kasnije obavlja weighted k-NN metodom.

### 3.2. Tok podataka

```text
STL-10
   |
   v
JPEG slike + metadata.csv
   |
   v
CLIP model
   |
   v
normalizovani 512D embedding vektori
   |
   v
Qdrant kolekcija stl10_clip_images
   |
   +--> similarity search i filter
   +--> CRUD demo
   +--> analiza grešaka
   `--> analiza kvaliteta dataseta
```

Kolekcija se kreira sa fiksnom dimenzijom 512 i Cosine metrikom. Pointovi se zatim uvoze paketno, u batch-evima od 1.000 pointova. Pre importa proveravaju se dimenzija vektora, potrebne metadata kolone, jedinstvenost ID-eva i validnost `embedding_index` vrednosti.

### 3.3. Pretraga sličnih slika

Za pretragu prema postojećoj slici aplikacija:

1. učita point prema ID-u;
2. uzme njegov vektor;
3. pošalje vektor kao Qdrant query;
4. isključi samu upitnu sliku iz rezultata;
5. prikaže `top-k` drugih slika, score i payload.

Opcioni `label` filter ograničava rezultate na traženu klasu. UI omogućava izbor HNSW ili exact pretrage, kao i podešavanje `hnsw_ef` parametra.

### 3.4. Evaluacija modela

Analiza koristi uravnotežen uzorak od **1.000 stvarnih označenih slika**, po 100 iz svake klase. Za svaku sliku Qdrant pronalazi pet najbližih drugih označenih slika. Sama upitna slika se obavezno izbacuje, što predstavlja leave-one-out postupak.

Svaki sused glasa za svoju klasu, a njegov Cosine score predstavlja težinu glasa. Klasa sa najvećim ukupnim ponderisanim glasom postaje predikcija. Za pogrešne odluke čuvaju se najbliži susedi i jednostavna dijagnoza, kao što su `class_confusion` ili `boundary_case`.

Uzorak od 1.000 slika izabran je zato što je:

- uravnotežen po klasama;
- sastavljen od stvarnih podataka;
- ponovljiv;
- dovoljno brz za interaktivnu demonstraciju.

Analitička skripta koristi Qdrant exact pretragu radi stabilnog poređenja i generiše CSV, JSON i HTML izveštaje, uključujući matricu konfuzije.

### 3.5. Provera kvaliteta dataseta

Druga analiza koristi isti uzorak od 1.000 slika i pronalazi jedinstvene parove čija je sličnost najmanje `0.94`.

| Kategorija | Prag |
|---|---:|
| veoma sličan par | `0.94` |
| verovatan duplikat | `0.95` |
| veoma verovatan duplikat | `0.97` |

Ista slika može pripadati većem broju parova. Zbog toga se povezani parovi spajaju u grupe. Za svaku grupu bira se reprezentativna slika, dok ostale dobijaju jednu od preporuka:

- `keep` – zadržati;
- `review` – ručno pregledati;
- `remove_candidate` – strogi kandidat za izostavljanje iz kopije.

Originalni dataset se ne menja. Ako korisnik potvrdi postupak, pravi se nova kopija u `data/cleaned/`. Parovi sa različitim labelama ne uklanjaju se automatski jer velika vektorska sličnost sama po sebi nije dokaz pogrešne anotacije.

## 4. Korisnički interfejs i pokretanje

Lokalni UI je napravljen bez dodatnog frontend framework-a. Python server u `ui/server.py` poslužuje HTML, CSS i JavaScript, komunicira sa Qdrantom i pokreće analitičke skripte.

Interfejs sadrži šest celina:

1. pregled stanja sistema;
2. similarity search i payload filter;
3. bezbedan CRUD demo;
4. evaluaciju modela;
5. proveru kvaliteta dataseta;
6. prezentacioni režim za odbranu.

Linearni deploy pokreće Qdrant, proverava embedding fajlove, kreira kolekciju, uvozi podatke, verifikuje import i tek zatim pokreće UI:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1
```

Ako neki korak ne uspe, sledeći koraci se ne izvršavaju. Za kasnije pokretanje, kada su podaci već uvezeni, dovoljno je pokrenuti Docker i `START_UI.bat`.

## 5. Testiranje i rezultati

Projekat sadrži unit testove logike i posebne testove stvarne baze. Testovi stvarnih podataka proveravaju:

- da broj Qdrant pointova odgovara metapodacima;
- da kolekcija koristi vektore dimenzije 512;
- da se stvarni Qdrant vektor slaže sa lokalnim embeddingom;
- da payload filter i similarity search vraćaju stvarne rezultate;
- da weighted k-NN radi nad stvarnim susedima.

Poslednji generisani izveštaji u projektu daju sledeće rezultate:

| Analiza | Rezultat |
|---|---|
| Weighted k-NN | 991 tačna od 1.000 predikcija |
| Tačnost | 99,10% |
| Greške | 9, od toga 8 `class_confusion` i 1 `boundary_case` |
| Slični parovi | 16 parova sa score vrednošću najmanje 0,94 |
| Grupe | 10 grupa sa ukupno 23 slike |
| Strogi kandidati | 2 slike za izostavljanje iz kopije |
| Potencijalna očišćena kopija | 998 slika |

## 6. Poređenje sa relacionom bazom

| Zahtev | Qdrant | Relaciona baza |
|---|---|---|
| Pretraga prema vektorskoj sličnosti | ugrađena i indeksirana | zahteva dodatnu ekstenziju ili ručno računanje |
| Čuvanje metadata podataka | payload | kolone u tabeli |
| Filtriranje prema labeli | payload filter i indeks | indeksirana kolona |
| Pronalaženje najbližih slika | HNSW ili exact query | nije osnovna operacija |
| Složene relacije i transakcioni podaci | nije glavni cilj projekta | prirodna primena relacionog modela |

Qdrant ne zamenjuje relacionu bazu u svakom slučaju. Njegova prednost u ovom projektu je što su centralni problem vektori i pretraga sličnosti. Za podatke sa složenim relacijama, velikim brojem transakcionih pravila i JOIN operacija relaciona baza bi i dalje bila prikladnija.

## 7. Ključne projektne odluke i ograničenja

Najvažnije projektne odluke su:

- korišćen je unapred trenirani CLIP da projekat ostane fokusiran na bazu i pretragu;
- svih 113.000 vektora čuvaju se u Qdrantu;
- interaktivne analize koriste 1.000 stvarnih slika radi brzine i ponovljivosti;
- za analize se koristi exact pretraga, dok je HNSW dostupan za standardnu pretragu i benchmark;
- originalni pointovi i originalni dataset zaštićeni su od automatskog brisanja;
- rezultati se generišu iz baze.

Ograničenja su:

- CLIP nije dodatno treniran za STL-10;
- weighted k-NN je jednostavan, objašnjiv klasifikator, a ne poseban neuronski model;
- pragovi za slične parove su heuristike i zahtevaju vizuelni pregled;
- rezultati dve analize odnose se na uzorak od 1.000 slika;
- deployment je lokalni single-node sistem bez autentifikacije.

## 8. Zaključak

Projekat pokazuje kompletan tok rada sa vektorskom bazom: od slike i embeddinga do indeksiranog pointa, similarity search upita i praktične analize rezultata. Qdrant je odgovarajući izbor jer objedinjuje čuvanje vektora, payload metadata podatke, filtriranje i pretragu najbližih suseda.

Pored osnovne pretrage, projekat pokazuje da se ista vektorska kolekcija može koristiti za weighted k-NN klasifikaciju, objašnjavanje grešaka i pronalaženje veoma sličnih slika.

## 9. Literatura

1. Qdrant, „Manage Data”: https://qdrant.tech/documentation/manage-data/
2. Qdrant, „Indexing”: https://qdrant.tech/documentation/manage-data/indexing/
3. Qdrant, „Local Quickstart”: https://qdrant.tech/documentation/quick-start/
4. Qdrant GitHub repozitorijum: https://github.com/qdrant/qdrant
5. Hugging Face, `openai/clip-vit-base-patch32`: https://huggingface.co/openai/clip-vit-base-patch32
6. STL-10 dataset: https://cs.stanford.edu/~acoates/stl10/
