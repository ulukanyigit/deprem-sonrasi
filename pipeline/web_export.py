
# web_export.py — web uygulaması için hafif veri
import os, sys, json, pickle, time, numpy as np, pandas as pd, geopandas as gpd, shapely
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.environ.get("DEPREM_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))); D = os.path.join(ROOT,"data","il"); W = os.path.join(ROOT,"web","data"); os.makedirs(W, exist_ok=True)
CRS = "EPSG:5254"
SC = [f"{s}{x}" for s in ("ibb","dogu","birlesik") for x in ("_hafif","","_kotumser")]
LOG = os.path.join(D,"web_log.txt")
def log(*a):
    m = time.strftime("[%H:%M:%S] ")+" ".join(map(str,a)); print(m, flush=True); open(LOG,"a",encoding="utf-8").write(m+"\n")
def r5(g): return shapely.set_precision(g, 0.00001)
def dump(name, obj):
    p = os.path.join(W, name); json.dump(obj, open(p,"w",encoding="utf-8"), ensure_ascii=False, separators=(",",":"), allow_nan=False, default=lambda o: (None if (hasattr(o,"item") and o!=o) else o.item()) if hasattr(o,"item") else str(o)); log(name, round(os.path.getsize(p)/1e6,2), "MB")
def coords(g):
    gj = shapely.geometry.mapping(g); return gj
open(LOG,"w").write("")
import faulthandler; faulthandler.enable(open(os.path.join(D,"web_fault.txt"),"w"))
log("başla")
R = pickle.load(open(os.path.join(D,"sim.pkl"),"rb"))
I = gpd.read_file(os.path.join(D,"ilceler.gpkg")); log("sim yüklendi")
# ---- binalar → mahalle (OSM) ve ilçe, senaryo hasar oranları
B = gpd.read_parquet(os.path.join(D,"binalar.parquet"), columns=["ilce","geometry"]); BA = pd.read_parquet(os.path.join(D,"bina_attr.parquet"))
log("binalar yüklendi", len(B)); cen = gpd.GeoDataFrame(geometry=B.centroid, crs=CRS); B = B.drop(columns="geometry"); import gc; gc.collect()
Mh = gpd.read_file(os.path.join(D,"osm_il.gpkg"), layer="mahalle"); Mh = Mh[Mh.admin_level=="8"][["name","geometry"]].reset_index(drop=True); Mh["mid"]=np.arange(len(Mh))
jm = gpd.sjoin(cen, Mh, how="left", predicate="within"); jm = jm[~jm.index.duplicated()]; log("mahalle eşleşti"); del cen
df = pd.DataFrame(dict(mid=jm.mid.values, ilce=B.ilce.values, pop=BA.nufus.values.astype(float)))
for s in ("ibb","dogu","birlesik"): df[f"hasar_{s}"] = (BA[f"p_cok_{s}"]+BA[f"p_agir_{s}"]).values
for k in SC:
    df[f"iso_{k}"]=R[k]["p_iso"]; df[f"cut_{k}"]=R[k]["p_cut"]
    Tt_load = R[k]["load"]
Tt = gpd.read_parquet(os.path.join(D,"toplanma.parquet"))
for k in SC:
    over = R[k]["load"] > Tt.kapasite.values; a = R["base_a"]
    df[f"tas_{k}"] = np.where(a>=0, over[np.clip(a,0,None)], False).astype(float)
df["hdk"] = np.where(np.isfinite(R["base_h"]), R["base_h"], np.nan); df["tm"] = np.where(np.isfinite(R["base_w"]), R["base_w"], np.nan)
def wm(g, c):
    w = g["pop"].values; v = g[c].values; m = np.isfinite(v); return float((v[m]*w[m]).sum()/max(w[m].sum(),1e-9))
log("bina tablosu hazır"); g = df[df.mid.notna()].groupby("mid")
feat=[]
Mh4 = Mh.to_crs(4326); Mh4["geometry"] = r5(Mh4.geometry.simplify(0.00012, preserve_topology=True).values)
for mid, gg in g:
    if len(gg) < 30: continue
    row = Mh4.iloc[int(mid)]
    props = dict(n=row["name"].replace(" Mahallesi",""), i=gg.ilce.mode().iat[0], b=int(len(gg)), p=int(round(gg["pop"].sum())),
        d={s: round(100*wm(gg,f"hasar_{s}"),1) for s in ("ibb","dogu","birlesik")},
        h=[round(100*wm(gg,f"iso_{k}"),1) for k in SC], t=[round(100*wm(gg,f"cut_{k}"),1) for k in SC], o=[round(100*wm(gg,f"tas_{k}"),0) for k in SC],
        hdk=round(np.nanmedian(gg.hdk),1) if gg.hdk.notna().any() else None, tm=int(np.nanmedian(gg.tm)) if gg.tm.notna().any() else None)
    feat.append(dict(type="Feature", properties=props, geometry=shapely.geometry.mapping(row.geometry)))
dump("mahalleler.json", dict(type="FeatureCollection", features=feat)); log("mahalle", len(feat))
# ---- yollar: herhangi bir senaryoda kapanma ≥ %5
E = gpd.read_parquet(os.path.join(D,"kenarlar.parquet"), columns=["hw","uzunluk","geometry"])
P = np.vstack([R[k]["p_blk"] for k in SC]).T
keep = P.max(1) >= 0.05
Ek = E[keep].copy(); Pk = np.round(100*P[keep]).astype(int)
# ilçe ataması — en yakın ilçe (Adalar düzeltmesi)
mid_pts = gpd.GeoDataFrame(geometry=Ek.representative_point(), crs=CRS)
jn = gpd.sjoin_nearest(mid_pts, I[["ilce","geometry"]], how="left", max_distance=3000); jn = jn[~jn.index.duplicated()]
Ek4 = Ek.to_crs(4326); geoms = r5(shapely.simplify(Ek4.geometry.values, 0.00003))
cs = [np.round(np.asarray(shapely.get_coordinates(gm)),5).ravel().tolist() for gm in geoms]
cls = Ek.hw.map({"motorway":3,"trunk":3,"primary":2,"secondary":2,"tertiary":1}).fillna(0).astype(int).tolist()
dump("yollar.json", dict(sc=SC, c=cs, v=Pk.tolist(), k=cls))
log("yol", int(keep.sum()), "/", len(E))
# ilçe kapalı km — tüm kenarlar, en yakın ilçe
allp = gpd.GeoDataFrame(geometry=E.representative_point(), crs=CRS)
ja = gpd.sjoin_nearest(allp, I[["ilce","geometry"]], how="left", max_distance=3000); ja = ja[~ja.index.duplicated()]
E["ilce"] = ja.ilce.values
# ---- toplanma alanları (nokta)
T4 = Tt.to_crs(4326); rp = r5(shapely.point_on_surface(T4.geometry.values))
dump("toplanma.json", dict(sc=SC, x=np.round(shapely.get_x(rp),5).tolist(), y=np.round(shapely.get_y(rp),5).tolist(),
     a=Tt.alan.round(0).astype(int).tolist(), k=Tt.kapasite.round(0).astype(int).tolist(), n=Tt.ad.fillna("").tolist(), t=Tt.tur.fillna("").tolist(),
     l=[np.round(R[k]["load"]).astype(int).tolist() for k in SC]))
# ---- hastaneler
H = gpd.read_parquet(os.path.join(D,"hastaneler.parquet")).to_crs(4326)
dump("hastaneler.json", dict(x=np.round(H.geometry.x,5).tolist(), y=np.round(H.geometry.y,5).tolist(), n=H.ad.tolist(), t=H.tur.tolist()))
# ---- ilçeler + özet
rows=[]; tot={}
for il, gg in df.groupby("ilce"):
    em = (E.ilce==il).values
    rows.append(dict(ilce=il, p=int(round(gg["pop"].sum())), d={s: round(100*wm(gg,f"hasar_{s}"),1) for s in ("ibb","dogu","birlesik")},
        h=[round(100*wm(gg,f"iso_{k}"),1) for k in SC], t=[round(100*wm(gg,f"cut_{k}"),1) for k in SC], o=[round(100*wm(gg,f"tas_{k}"),0) for k in SC],
        km=[round(float((R[k]["p_blk"][em]*E.uzunluk.values[em]).sum()/1000),1) for k in SC]))
S = {r["ilce"]: r for r in rows}
I4 = I.to_crs(4326); I4["geometry"] = r5(I4.geometry.simplify(0.0004, preserve_topology=True).values)
dump("ilceler.json", dict(type="FeatureCollection", features=[dict(type="Feature", properties=S.get(r.ilce, dict(ilce=r.ilce)), geometry=shapely.geometry.mapping(r.geometry)) for r in I4.itertuples()]))
pop = df["pop"].values
summ = {k: dict(hastane_kopuk=int(R[k]["stats"].izole_nufus.mean()), toplanma_kopuk=int(R[k]["stats"].toplanma_kopuk_nufus.mean()),
       kapali_km=round(float(R[k]["stats"].kapali_km.mean())), tasan=round(100*float((df[f"tas_{k}"]*pop).sum()/pop.sum()),1),
       p5=int(np.percentile(R[k]["stats"].izole_nufus,5)), p95=int(np.percentile(R[k]["stats"].izole_nufus,95))) for k in SC}
dump("ozet.json", dict(sc=SC, nufus=int(pop.sum()), bina=int(len(df)), yol_km=round(float(E.uzunluk.sum()/1000)), toplanma=len(Tt), kapasite=int(Tt.kapasite.sum()), hastane=len(H), s=summ))
log("BİTTİ")
