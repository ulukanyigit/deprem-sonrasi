
# senaryo_model.py — fay kırılması + yer sarsıntısı (BSSA14 biçimi) + İBB'ye kalibre kırılganlık
import os, sys, re, time, unicodedata, json, numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely.geometry import LineString
from scipy.stats import norm as N
from scipy.optimize import minimize
ROOT = os.environ.get("DEPREM_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))); D = os.path.join(ROOT,"data","il"); EV = os.environ.get("EV_KARNESI_ROOT", os.path.join(ROOT, "..", "EV KARNESİ"))
CRS = "EPSG:5254"; BETA = 0.6
LOG = os.path.join(D,"senaryo_log.txt")
def log(*a):
    m = time.strftime("[%H:%M:%S] ") + " ".join(str(x) for x in a); print(m, flush=True)
    open(LOG,"a",encoding="utf-8").write(m+"\n")

# Ana Marmara Fayı — yaklaşık iz (lon, lat); ±5 km belirsizlik
FAY = [(27.55,40.82),(27.95,40.81),(28.25,40.84),(28.55,40.86),(28.85,40.87),(29.05,40.83),(29.25,40.77),(29.40,40.72)]
def iz(lon0, lon1):
    pts = [p for p in FAY if lon0-1e-9 <= p[0] <= lon1+1e-9]
    return gpd.GeoSeries([LineString(pts)], crs="EPSG:4326").to_crs(CRS).iloc[0]
KIRILMALAR = {
    "orta_marmara":     dict(lon=(27.95,28.85), ad="Orta Marmara (Kumburgaz)"),
    "kumburgaz_adalar": dict(lon=(28.25,29.40), ad="Kumburgaz + Adalar"),
    "silivri_tuzla":    dict(lon=(27.55,29.40), ad="Silivri açığı – Tuzla (tümü)"),
    "adalar":           dict(lon=(28.85,29.40), ad="Adalar segmenti"),
}

def ln_pga(M, rjb):
    e1, e4, e5, e6, Mh = 0.4856, 1.431, 0.05053, -0.1662, 5.5
    c1, c2, c3, Mref, h = -1.134, 0.1917, -0.008088, 4.5, 4.5
    FE = e1 + (e4*(M-Mh) + e5*(M-Mh)**2 if M <= Mh else e6*(M-Mh))
    R = np.sqrt(rjb**2 + h**2)
    return FE + (c1 + c2*(M-Mref))*np.log(R) + c3*(R-1)

def norm_ad(s):
    if s is None or (isinstance(s,float) and np.isnan(s)): return ""
    s = str(s).replace("İ","I").replace("ı","i").upper()
    s = unicodedata.normalize("NFKD", s); s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\b(MAHALLESI|MAHALLE|MAH)\b","",s); return re.sub(r"[^A-Z0-9]","",s)

def mahalleler():
    out = os.path.join(D,"mahalle_konum.parquet")
    S = pd.read_csv(os.path.join(EV,"data","raw","ibb_deprem_senaryo.csv"), sep=";", encoding="cp1254")
    T = pd.read_csv(os.path.join(EV,"data","raw","ibb_mahalle_bina.csv"), sep=";", encoding="cp1254")
    S = S.merge(T, left_on="mahalle_koy_uavt", right_on="mahalle_uavt", suffixes=("","_t"))
    S["toplam"] = S[["1980_oncesi","1980-2000_arasi","2000_sonrasi"]].sum(axis=1)
    S = S[S.toplam>0].copy()
    S["p_cok"] = S.cok_agir_hasarli_bina_sayisi/S.toplam
    S["p_ag"] = (S.cok_agir_hasarli_bina_sayisi+S.agir_hasarli_bina_sayisi)/S.toplam
    S["k_ilce"] = S.ilce_adi.map(norm_ad).replace({"EYUP":"EYUPSULTAN"})
    S["key"] = S.k_ilce + "|" + S.mahalle_adi.map(norm_ad)
    if not os.path.exists(out):
        B = gpd.read_parquet(os.path.join(D,"binalar.parquet"), columns=["ilce","mah","geometry"])
        c = shapely.centroid(B.geometry.values)
        X = pd.DataFrame(dict(key=B.ilce.map(norm_ad).values+"|"+B.mah.map(norm_ad).values, x=shapely.get_x(c), y=shapely.get_y(c)))
        X.to_parquet(out)
    X = pd.read_parquet(out)
    xy = X.groupby("key")[["x","y"]].median()
    S = S.merge(xy, left_on="key", right_index=True, how="left")
    il_xy = X.assign(k=X.key.str.split("|").str[0]).groupby("k")[["x","y"]].median()
    miss = S.x.isna(); S.loc[miss,"x"] = S.loc[miss,"k_ilce"].map(il_xy.x); S.loc[miss,"y"] = S.loc[miss,"k_ilce"].map(il_xy.y)
    S["konum"] = np.where(miss, "ilce_merkezi", "mahalle")
    return S[S.x.notna()].reset_index(drop=True)

def rjb(S, line):
    return shapely.distance(shapely.points(S.x.values, S.y.values), line)/1000.0

def sinif_modeli(S, lnA):
    W = S[["1980_oncesi","1980-2000_arasi","2000_sonrasi"]].values / S.toplam.values[:,None]
    H = S[["5-9 kat_arasi","9-19 kat_arasi"]].values.sum(1) / S.toplam.values
    y = S.p_ag.values; w = S.toplam.values
    def pred(th): return (W * N.cdf((lnA[:,None]-(th[:3]+th[3]*H[:,None]))/BETA)).sum(1)
    f = lambda th: np.sum(w*(pred(th)-y)**2)
    best=None
    for x0 in ([lnA.mean()+0.5, lnA.mean()+1.0, lnA.mean()+1.5, 0.0], [lnA.mean()+1.5, lnA.mean()+2.0, lnA.mean()+2.5, -0.5]):
        r = minimize(f, x0=np.array(x0), method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-5, fatol=1e-12))
        if best is None or r.fun < best.fun: best = r
    p = pred(best.x); r2 = 1 - np.sum(w*(p-y)**2)/np.sum(w*(y-np.average(y,weights=w))**2)
    return r2, best.x, p

if __name__ == "__main__":
    open(LOG,"w",encoding="utf-8").write("")
    S = mahalleler(); log("mahalle", len(S), S.konum.value_counts().to_dict())
    rows=[]
    for k,v in KIRILMALAR.items():
        line = iz(*v["lon"]); L = line.length/1000
        for M in (7.2, 7.5):
            r2, th, p = sinif_modeli(S, ln_pga(M, rjb(S,line)))
            rows.append(dict(kirilma=k, ad=v["ad"], uzunluk_km=round(L,1), Mw=M, R2=round(r2,3)))
            log(f"{v['ad']:30s} {L:6.1f} km  Mw{M}  R²={r2:.3f}  eşik={np.round(th,2)}")
    # mesafe tek başına — referans
    for k,v in KIRILMALAR.items():
        d = rjb(S, iz(*v["lon"])); w=S.toplam.values
        c = np.corrcoef(np.log(np.sqrt(d**2+20)), np.log(S.p_ag.values+1e-3))[0,1]
        log(f"  {k}: log(mesafe)–log(hasar) korelasyonu {c:.2f}")
    pd.DataFrame(rows).to_csv(os.path.join(D,"kirilma_secimi.csv"), index=False, encoding="utf-8-sig")
    S.to_parquet(os.path.join(D,"mahalle_senaryo_girdi.parquet")); log("BİTTİ")
