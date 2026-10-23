# Yol Kapanırsa — Deprem sonrası İstanbul

**[Canlı harita / Live map → ulukanyigit.github.io/deprem-sonrasi](https://ulukanyigit.github.io/deprem-sonrasi/)**

[Türkçe](#türkçe) · [English](#english)

---

## Türkçe

İstanbul'da büyük bir depremden sonra yıkılan binaların enkazı hangi sokakları kapatır, kimler hastaneye araçla ve toplanma alanına yürüyerek ulaşamaz? Sonuçlar eğik 3D bir kent maketi üzerinde gösteriliyor: uzaktan mahalle sütunları, yakınlaşınca gerçek yükseklikleriyle 1,08 milyon bina. Bu proje soruyu il genelinde, **1,08 milyon bina**, **447 bin yol parçası**, **528 hastane** ve **11.199 toplanma alanı** üzerinde Monte Carlo simülasyonuyla yanıtlıyor.

### Öne çıkan bulgular (olası şiddet, 100 tur ortalaması)

| Senaryo | Hastaneyle araç bağlantısı kopan | Toplanma alanına yürüyemeyen | Kapanan yol |
|---|---|---|---|
| Orta Marmara, Mw 7,5 (İBB senaryosu) | 408.212 kişi (%2,6) | 163.231 kişi | 944 km |
| Adalar segmenti, Mw 7,1 | 356.175 kişi (%2,3) | 141.547 kişi | 970 km |
| Kumburgaz + Adalar, Mw 7,4 | 529.124 kişi (%3,4) | 208.666 kişi | 1.359 km |

- Fatih, Bakırköy ve Zeytinburnu her senaryoda en kırılgan ilçeler arasında. Dar sokak dokusu, deprem nereden gelirse gelsin risk yaratıyor.
- İBB senaryosunda Anadolu yakası hafif kalıyor. Bunun nedeni, İBB'nin mahalle hasar dağılımını en iyi açıklayan kırılmanın Adalar'ın batısında biten Orta Marmara segmenti olması (R² 0,67). Doğu kırılmasında Anadolu yakasında kapanan yol 154 km'den 485 km'ye çıkıyor.
- Toplanma alanlarının toplam kapasitesi nüfusun yaklaşık 5 katı, ama nüfusun %65'inin en yakın alanı gelen kişileri almıyor. Alan var, ama doğru yerde değil.

### Yöntem

1. **Bina stoku.** OSM bina tabanları ([Ev Karnesi](https://ulukanyigit.github.io/ev-karnesi/) projesinden), OSM'de olmayanlar Overture Maps (Microsoft ML Buildings) ile tamamlandı. Eksik kat sayıları İBB'nin mahalle bazlı kat dağılımına göre atandı.
2. **Hasar olasılığı.** İBB'nin mahalle bazlı deprem senaryosundaki ağır ve çok ağır hasarlı bina oranları. Doğu ve birleşik senaryolar için her mahallenin zemin ve bina zayıflığı İBB sonucundan geri hesaplandı; yalnızca sarsıntı değiştirildi:
   `p_yeni = Φ( Φ⁻¹(p_İBB) + Δln(PGA) / β )`, BSSA14 yer hareketi biçimi, β = 0,75 (veriden 0,51–0,75), faya en az 10 km.
3. **Enkaz.** Çok ağır hasarlı binanın yüksekliğinin %50'si, ağır hasarlının %15'i kadar yola taşar. Her yol 10 m'lik dilimlere bölünür, iki yandan gelen enkaz taşıt yolu genişliğinden düşülür. Net genişlik 3 m'nin altında araç, 1 m'nin altında yaya geçemez.
4. **Erişim.** Her turda kapalı yollar ağdan çıkarılır, tüm hastanelerden ve toplanma alanlarından çok kaynaklı Dijkstra çalıştırılır. Hastanelerin hasar görmediği varsayıldı.
5. **Toplanma alanları.** Resmi liste açık veri değil. Parklar, meydanlar, okul bahçeleri ve spor alanlarından, çevredeki her binanın yüksekliği kadar güvenlik payı çıkarılarak türetildi (en az 300 m², 10 m genişlik, kişi başı 1,5 m²).
6. **Nüfus.** TÜİK 2025 ilçe nüfusları binalara kat alanıyla orantılı dağıtıldı.

### Sınırlamalar

- Bu bir senaryo modelidir; tek tek binalar için hasar öngörüsü değildir.
- Fay izi yaklaşıktır (±5 km). Zeminin büyük sarsıntılardaki doğrusal olmayan davranışı modellenmedi.
- Toplanma alanı taşması en yakın alana gidiş varsayımına dayanır; bir üst sınırdır.
- Adalar'da ulaşım vapura bağlıdır; oradaki sonuçlar gösterge niteliğindedir.

---

## English

After a major earthquake in Istanbul, which streets will be blocked by debris from collapsed buildings, and who will lose access to hospitals by car or to an assembly area on foot? Results are shown on a tilted 3D city model: neighbourhood columns from afar, and all 1.08 million buildings at their real heights when you zoom in. This project answers that question city-wide with a Monte Carlo simulation over **1.08 million buildings**, **447,000 road segments**, **528 hospitals** and **11,199 assembly areas**.

### Key findings (likely severity, mean of 100 runs)

| Scenario | People cut off from hospitals by car | Cannot walk to an assembly area | Roads blocked |
|---|---|---|---|
| Central Marmara, Mw 7.5 (IMM scenario) | 408,212 (2.6%) | 163,231 | 944 km |
| Princes' Islands segment, Mw 7.1 | 356,175 (2.3%) | 141,547 | 970 km |
| Kumburgaz + Princes' Islands, Mw 7.4 | 529,124 (3.4%) | 208,666 | 1,359 km |

- Fatih, Bakırköy and Zeytinburnu stay among the most vulnerable districts in every scenario: their narrow street fabric is risky wherever the rupture is.
- The Asian side looks mild in the official IMM (İBB) scenario because the rupture that best explains IMM's neighbourhood damage pattern is the Central Marmara segment, which ends west of the Princes' Islands (R² 0.67). In an eastern rupture, blocked roads on the Asian side rise from 154 km to 485 km.
- Total assembly-area capacity is about five times the population, yet for 65% of residents the nearest area is over capacity. The space exists, but not where people live.

### Method

1. **Building stock.** OSM footprints (from the [Ev Karnesi](https://ulukanyigit.github.io/ev-karnesi/) project), gaps filled with Overture Maps (Microsoft ML Buildings). Missing storey counts assigned from IMM's neighbourhood storey distribution.
2. **Damage probability.** Heavy and very heavy damage rates from IMM's neighbourhood earthquake scenario. For the eastern and combined scenarios each neighbourhood's site and building vulnerability is back-calculated from the IMM result and only the shaking is changed:
   `p_new = Φ( Φ⁻¹(p_IMM) + Δln(PGA) / β )`, BSSA14 ground-motion form, β = 0.75 (0.51–0.75 estimated from the data), fault distance floored at 10 km.
3. **Debris.** A very heavily damaged building spills 50% of its height onto the road, a heavily damaged one 15%. Roads are split into 10 m bins and debris from both sides is subtracted from the carriageway width. Below 3 m clear width vehicles cannot pass; below 1 m pedestrians cannot.
4. **Access.** In every run blocked edges are removed and multi-source Dijkstra is run from all hospitals and assembly areas. Hospitals are assumed undamaged.
5. **Assembly areas.** The official list is not open data. Areas are derived from parks, squares, school yards and sports grounds, minus a safety buffer equal to the height of each surrounding building (min. 300 m², 10 m wide, 1.5 m² per person).
6. **Population.** TÜİK 2025 district populations distributed to buildings by floor area.

### Limitations

- This is a scenario model, not a damage forecast for individual buildings.
- Fault traces are approximate (±5 km). Non-linear site response under strong shaking is not modelled.
- Assembly-area overload assumes everyone walks to the nearest area, so it is an upper bound.
- The Princes' Islands depend on ferries; results there are indicative only.

---

### Repository / Depo

| Path | |
|---|---|
| `index.html` | Web app (MapLibre GL JS) |
| `data/` | Simplified results for the web app / Web uygulaması verisi |
| `data/bina/` | 3D building layer, one file per district, loaded on zoom / 3D bina katmanı, ilçe başına bir dosya |
| `pipeline/il_pipeline.py` | Full city pipeline: buildings → network → debris pairs → scenarios → assembly areas → Monte Carlo → outputs |
| `pipeline/senaryo_model.py`, `senaryo_beta.py` | Rupture selection and fragility (β) calibration against the IMM scenario |
| `pipeline/web_export.py`, `web_bina_export.py` | Export `data/` and `data/bina/` |
| `sonuclar/` | District summary, rupture selection and β sensitivity tables |

Run with Python 3.12, GeoPandas, Shapely 2, SciPy, DuckDB, GDAL. Set `DEPREM_ROOT` and `EV_KARNESI_ROOT` to the data folders.

### Data sources / Veri kaynakları

- İBB Açık Veri: deprem senaryosu analiz sonuçları, mahalle bina sayıları, sağlık kurumları
- © OpenStreetMap contributors (ODbL) — roads, buildings, open spaces
- Overture Maps Foundation — buildings
- Ergintav et al. (2014), *Istanbul's earthquake hot spots*, GRL — Princes' Islands segment
- Boore, Stewart, Seyhan & Atkinson (2014), NGA-West2 GMPE (functional form)
- TÜİK ADNKS 2025 district populations

Derived map data in `data/` is based on OpenStreetMap and is available under the ODbL.

---

Hamza Yiğit Ulukan · Geomatics engineer · [LinkedIn](https://www.linkedin.com/in/yulukan)
