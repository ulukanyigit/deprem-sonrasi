
# web_bina_export.py — 3D bina katmanı: ilçe başına bir dosya, yakınlaşınca yüklenir
import os, sys, json, pickle, time, re, unicodedata, numpy as np, pandas as pd, geopandas as gpd, shapely
ROOT = os.environ.get("DEPREM_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
D = os.path.join(ROOT,"data","il"); W = os.path.join(ROOT,"web","data","bina"); os.makedirs(W, exist_ok=True)
CRS = "EPSG:5254"; SC = [f"{s}{x}" for s in ("ibb","dogu","birlesik") for x in ("_hafif","","_kotumser")]
LOG = os.path.join(D,"web_bina_log.txt"); open(LOG,"w").write("")
def log(*a):
    m = time.strftime("[%H:%M:%S] ")+" ".join(map(str,a)); print(m, flush=True); open(LOG,"a",encoding="utf-8").write(m+"\n")
def slug(s):
    s = str(s).replace("İ","I").replace("ı","i").lower(); s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-z0-9]","", "".join(c for c in s if not unicodedata.combining(c)))
R = pickle.load(open(os.path.join(D,"sim.pkl"),"rb")); log("sim")
B = gpd.read_parquet(os.path.join(D,"binalar.parquet"), columns=["ilce","h","geometry"])
BA = pd.read_parquet(os.path.join(D,"bina_attr.parquet"))
Tt = pd.read_parquet(os.path.join(D,"toplanma.parquet"), columns=["kapasite"])
ISO = np.round(100*np.vstack([R[k]["p_iso"] for k in SC]).T).astype(np.int16)
CUT = np.round(100*np.vstack([R[k]["p_cut"] for k in SC]).T).astype(np.int16)
HAS = np.round(100*np.vstack([(BA[f"p_cok_{s}"]+BA[f"p_agir_{s}"]).values for s in ("ibb","dogu","birlesik")]).T).astype(np.int16)
a = R["base_a"]; TAS = np.zeros(len(B), np.int32)
for j,k in enumerate(SC):
    over = R[k]["load"] > Tt.kapasite.values
    TAS |= (np.where(a>=0, over[np.clip(a,0,None)], False).astype(np.int32) << j)
H = np.round(np.clip(B.h.fillna(6).values, 2, 200)).astype(int)
log("öznitelikler hazır")
index = []
for il, ix in B.groupby("ilce").indices.items():
    g = gpd.GeoSeries(B.geometry.values[ix], crs=CRS).simplify(0.4).to_crs(4326)
    xs, ys, rings = [], [], []
    for geom in g.values:
        if geom is None or geom.is_empty: rings.append(None); continue
        poly = max(shapely.get_parts(geom), key=lambda p: p.area) if geom.geom_type != "Polygon" else geom
        c = np.round(np.asarray(poly.exterior.coords)[:-1]*1e6).astype(np.int64)
        rings.append(c)
    allc = np.vstack([r for r in rings if r is not None]); x0, y0 = int(allc[:,0].min()), int(allc[:,1].min())
    feats_c, feats_p = [], []
    for k, r in enumerate(rings):
        if r is None or len(r) < 3: continue
        rr = r - [x0, y0]; d = np.diff(np.vstack([[0,0], rr]), axis=0).ravel().tolist()
        i = ix[k]
        iso = ISO[i].tolist(); cut = CUT[i].tolist()
        feats_c.append(d)
        feats_p.append([int(H[i]), HAS[i].tolist(), iso if any(iso) else 0, cut if any(cut) else 0, int(TAS[i])])
    key = slug(il)
    obj = dict(x0=x0, y0=y0, c=feats_c, p=feats_p)
    fp = os.path.join(W, f"{key}.json"); json.dump(obj, open(fp,"w",encoding="utf-8"), separators=(",",":"))
    b = g.total_bounds
    index.append(dict(ilce=il, f=f"{key}.json", n=len(feats_c), bbox=[round(float(v),5) for v in b], mb=round(os.path.getsize(fp)/1e6,2)))
    log(il, len(feats_c), "bina", index[-1]["mb"], "MB")
json.dump(dict(sc=SC, ilceler=index), open(os.path.join(W,"index.json"),"w",encoding="utf-8"), ensure_ascii=False, separators=(",",":"))
log("toplam MB", round(sum(x["mb"] for x in index),1)); log("BİTTİ")
