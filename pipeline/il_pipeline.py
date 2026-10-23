
# il_pipeline.py — İstanbul il geneli deprem sonrası yol kapanması, hastane ve toplanma alanı erişimi
# Çalıştırma: python il_pipeline.py [adim]   (adımlar ara sonuç kaydeder; tekrar çalıştırınca kaldığı yerden devam eder)
import os, sys, json, re, time, gc, unicodedata, pickle
import numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely.geometry import Polygon, LineString
from scipy.spatial import cKDTree
import scipy.sparse as sp, scipy.sparse.csgraph as cg
from collections import Counter

ROOT = os.environ.get("DEPREM_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))); EV = os.environ.get("EV_KARNESI_ROOT", os.path.join(ROOT, "..", "EV KARNESİ"))
D = os.path.join(ROOT,"data","il"); CRS = "EPSG:5254"
T0 = time.time()
def log(*a):
    msg = f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:6.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(os.path.join(D,"log.txt"),"a",encoding="utf-8") as f: f.write(msg+"\n")

# 31.12.2025 ADNKS (Wikipedia: List of districts of Istanbul)
NUFUS = {"Adalar":17489,"Arnavutköy":358469,"Ataşehir":412125,"Avcılar":440663,"Bağcılar":707635,"Bahçelievler":539035,"Bakırköy":218204,
 "Başakşehir":536797,"Bayrampaşa":272978,"Beşiktaş":165895,"Beykoz":246833,"Beylikdüzü":422988,"Beyoğlu":215991,"Büyükçekmece":283239,
 "Çatalca":81143,"Çekmeköy":315959,"Esenler":419878,"Esenyurt":1003905,"Eyüpsultan":425216,"Fatih":351786,"Gaziosmanpaşa":478395,
 "Güngören":251242,"Kadıköy":458573,"Kağıthane":446420,"Kartal":475630,"Küçükçekmece":785270,"Maltepe":525044,"Pendik":752033,
 "Sancaktepe":507500,"Sarıyer":344883,"Silivri":240029,"Sultanbeyli":378908,"Sultangazi":529306,"Şile":50090,"Şişli":261959,
 "Tuzla":313865,"Ümraniye":728913,"Üsküdar":514294,"Zeytinburnu":275471}

# ---- model parametreleri
K_FULL, K_PART = 0.5, 0.15; BIN = 10.0; MIN_CLEAR = 3.0; WALK_CLEAR = 1.0
SPEED = {"motorway":40,"trunk":30,"primary":30,"secondary":25,"tertiary":20}; DEFAULT_SPEED = 15
M2_KISI = 1.5; YAKIN_M = 500; MIN_COMP = 50; N_RUNS = int(os.environ.get("N_RUNS","100"))
SCEN = [(f"{src}{suf}", src, m) for src in ("ibb","dogu","birlesik") for suf, m in (("_hafif",0.5),("",1.0),("_kotumser",1.5))]

def norm(s):
    if s is None or (isinstance(s,float) and np.isnan(s)): return ""
    s = str(s).replace("İ","I").replace("ı","i").upper()
    s = unicodedata.normalize("NFKD", s); s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\b(MAHALLESI|MAHALLE|MAH)\b","",s); return re.sub(r"[^A-Z0-9]","",s)

def keys():
    return sorted(f[:-5] for f in os.listdir(os.path.join(EV,"ev-karnesi-repo","data")) if f.endswith(".json"))


OSM_IL = os.path.join(D,"osm_il.gpkg")
def s0_osm():
    if os.path.exists(OSM_IL+".ok"): return
    from osgeo import gdal
    gdal.UseExceptions(); gdal.SetConfigOption("OSM_USE_CUSTOM_INDEXING","NO"); gdal.SetConfigOption("OGR_INTERLEAVED_READING","YES")
    pbf = os.path.join(EV,"data","raw","turkey-latest.osm.pbf")
    I = gpd.read_file(os.path.join(D,"ilceler.gpkg")).to_crs(4326); x0,y0,x1,y1 = I.total_bounds; x0-=0.02; y0-=0.02; x1+=0.02; y1+=0.02
    if os.path.exists(OSM_IL): os.remove(OSM_IL)
    jobs = [
     ("roads","lines","SELECT osm_id, name, highway, other_tags FROM lines WHERE highway IN ('motorway','motorway_link','trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','residential','unclassified','living_street','service','road')"),
     ("mahalle","multipolygons","SELECT osm_id, name, admin_level FROM multipolygons WHERE boundary='administrative' AND admin_level IN ('6','8')"),
     ("acik_alan_aday","multipolygons","""SELECT osm_id, osm_way_id, name, leisure, landuse, amenity, place, military, other_tags FROM multipolygons WHERE
       leisure IN ('park','garden','playground','pitch','recreation_ground','sports_centre','stadium','track','common')
       OR landuse IN ('grass','recreation_ground','village_green','cemetery','meadow')
       OR amenity IN ('school','university','college','kindergarten','grave_yard')
       OR place='square' OR other_tags LIKE '%"highway"=>"pedestrian"%' OR other_tags LIKE '%"emergency"=>"assembly_point"%'"""),
    ]
    for i,(name,lyr,sql) in enumerate(jobs):
        gdal.VectorTranslate(OSM_IL, pbf, options=gdal.VectorTranslateOptions(format="GPKG", SQLStatement=sql, layerName=name,
            accessMode=None if i==0 else "update", dstSRS="EPSG:5254", spatFilter=[x0,y0,x1,y1]))
        log("osm_il", name, "yazıldı")
    open(OSM_IL+".ok","w").write("ok")

# ================= 1. binalar
def s1_binalar():
    out = os.path.join(D,"binalar_ek.parquet")
    if os.path.exists(out): return
    parts=[]; borders=[]
    for k in keys():
        d = json.load(open(os.path.join(EV,"ev-karnesi-repo","data",f"{k}.json"), encoding="utf-8"))
        rows=[]
        for b in d["b"]:
            try: rows.append((f"{k}_{b['id']}", d["name"], b.get("m"), b.get("lv"), b.get("h"), b.get("y"), Polygon(b["r"])))
            except Exception: pass
        g = gpd.GeoDataFrame(pd.DataFrame(rows, columns=["bid","ilce","mahalle","kat","h","yil","geometry"]), geometry="geometry", crs="EPSG:4326").to_crs(CRS)
        parts.append(g); borders.append((d["name"], Polygon(d["border"][0])))
        log("bina", k, len(g)); del d
    B = pd.concat(parts, ignore_index=True); B["kaynak"]="ev_karnesi"
    B["geometry"] = shapely.make_valid(B.geometry.values)
    B = B[B.geometry.geom_type.isin(["Polygon","MultiPolygon"])].reset_index(drop=True)
    B["h"] = B["h"].fillna(B["kat"].fillna(3)*3.0)
    B.to_parquet(out)
    gpd.GeoDataFrame(pd.DataFrame(borders, columns=["ilce","geometry"]), geometry="geometry", crs="EPSG:4326").to_crs(CRS).to_file(os.path.join(D,"ilceler.gpkg"), driver="GPKG")
    log("binalar toplam", len(B))

# ================= 2. Overture
def s2_overture():
    out = os.path.join(D,"overture.parquet")
    if os.path.exists(out): return
    import duckdb
    I = gpd.read_file(os.path.join(D,"ilceler.gpkg")).to_crs(4326); x0,y0,x1,y1 = I.total_bounds
    con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2'; SET threads=8;")
    tmp = out+".tmp"
    con.execute(f"""COPY (SELECT id, height, num_floors, sources[1].dataset AS kaynak1, geometry AS geom
      FROM read_parquet('s3://overturemaps-us-west-2/release/2026-10-21.0/theme=buildings/type=building/*', hive_partitioning=1)
      WHERE bbox.xmin < {x1} AND bbox.xmax > {x0} AND bbox.ymin < {y1} AND bbox.ymax > {y0}
        AND sources[1].dataset <> 'OpenStreetMap') TO '{tmp.replace(chr(92),'/')}' (FORMAT PARQUET)""")
    os.replace(tmp, out); log("overture (OSM dışı) indirildi", con.execute(f"SELECT count(*) FROM '{out.replace(chr(92),'/')}'").fetchone())

# ================= 3. tamamlama
def s3_tamamla():
    out = os.path.join(D,"binalar.parquet")
    if os.path.exists(out): return
    B = gpd.read_parquet(os.path.join(D,"binalar_ek.parquet"))
    I = gpd.read_file(os.path.join(D,"ilceler.gpkg"))
    O = pd.read_parquet(os.path.join(D,"overture.parquet"))
    geom = shapely.from_wkb(O.geom.values) if O.geom.dtype==object else shapely.from_wkb(O.geom.values)
    O = gpd.GeoDataFrame(O.drop(columns="geom"), geometry=geom, crs="EPSG:4326").to_crs(CRS)
    O["geometry"] = shapely.make_valid(O.geometry.values)
    O = O[O.geometry.geom_type.isin(["Polygon","MultiPolygon"]) & (O.area>=25)].reset_index(drop=True)
    log("overture aday", len(O))
    oi, bi = B.sindex.query(O.geometry, predicate="intersects")
    cov = np.zeros(len(O))
    for s in range(0, len(oi), 2_000_000):
        a = shapely.area(shapely.intersection(O.geometry.values[oi[s:s+2_000_000]], B.geometry.values[bi[s:s+2_000_000]]))
        cov += np.bincount(oi[s:s+2_000_000], weights=a, minlength=len(O))
    N = O[cov/O.area.values < 0.3].copy(); del O; gc.collect()
    cB = np.c_[shapely.get_x(shapely.centroid(B.geometry.values)), shapely.get_y(shapely.centroid(B.geometry.values))]
    tree = cKDTree(cB); cN = np.c_[N.centroid.x, N.centroid.y]
    katB = B.kat.fillna(3).values; kat_nb = np.full(len(N), np.nan)
    for s in range(0, len(N), 200000):
        nb = tree.query_ball_point(cN[s:s+200000], 60)
        kat_nb[s:s+200000] = [np.median(katB[ix]) if len(ix)>=3 else np.nan for ix in nb]
    kat = N.num_floors.astype(float).values
    kat = np.where(np.isnan(kat) & N.height.notna().values, np.round(N.height.values/3.2), kat)
    kat = np.where(np.isnan(kat), kat_nb, kat); kat = np.clip(np.where(np.isnan(kat),2,kat),1,40)
    kat = np.where(N.area.values<60, np.minimum(kat,2), kat)
    N["kat"]=kat; N["h"]=np.where(N.height.notna(), N.height, kat*3.2)
    j = gpd.sjoin(gpd.GeoDataFrame(geometry=N.centroid, crs=CRS), I, how="left", predicate="within"); j=j[~j.index.duplicated()]
    N["ilce"]=j.ilce.values; N=N[N.ilce.notna()]
    N["bid"]="ov_"+N["id"].astype(str); N["mahalle"]=None; N["yil"]=np.nan; N["kaynak"]="overture_"+N.kaynak1.astype(str)
    cols=["bid","ilce","mahalle","kat","h","yil","kaynak","geometry"]
    M = pd.concat([B[cols], N[cols]], ignore_index=True)
    log("eklenen", len(N), "| ilçeye göre en çok:", N.ilce.value_counts().head(5).to_dict())
    # mahalle eksikse OSM mahalle sınırından
    Mh = gpd.read_file(OSM_IL, layer="mahalle"); Mh = Mh[Mh.admin_level=="8"][["name","geometry"]]
    j = gpd.sjoin(gpd.GeoDataFrame(geometry=M.centroid, crs=CRS), Mh, how="left", predicate="within"); j=j[~j.index.duplicated()]
    M["mah"] = M.mahalle.where(M.mahalle.notna() & (M.mahalle!=""), j["name"].values)
    M["mah"] = M.mah.fillna("").str.replace(r"\s*Mahallesi$","",regex=True)
    M.to_parquet(out); log("binalar (tamamlanmış)", len(M))


# ================= 3b. eklenen binaların katı — İBB mahalle kat dağılımına göre
def s3b_kat():
    mark = os.path.join(D,"kat_duzeltme.ok")
    if os.path.exists(mark): return
    fp = os.path.join(D,"binalar.parquet")
    B = gpd.read_parquet(fp)
    if not os.path.exists(os.path.join(D,"binalar_once_kat.parquet")): B.to_parquet(os.path.join(D,"binalar_once_kat.parquet"))
    T = pd.read_csv(os.path.join(EV,"data","raw","ibb_mahalle_bina.csv"), sep=";", encoding="cp1254")
    T["key"] = T.ilce_adi.map(norm).replace({"EYUP":"EYUPSULTAN"}) + "|" + T.mahalle_adi.map(norm)
    T = T.groupby("key")[["1-4 kat_arasi","5-9 kat_arasi","9-19 kat_arasi"]].sum()
    B["key"] = B.ilce.map(norm) + "|" + B.mah.map(norm)
    added = B.kaynak.str.startswith("overture").values; area = B.area.values
    cls = np.where(B.kat.values>=9, 2, np.where(B.kat.values>=5, 1, 0))
    ex = pd.crosstab(B.key[~added], cls[~added]).reindex(columns=[0,1,2], fill_value=0)
    kat_new = B.kat.values.astype(float).copy()
    target = added & (area >= 60)
    n_ok = n_miss = 0
    for key, ix in B[target].groupby("key").indices.items():
        ix = np.flatnonzero(target)[ix]
        if key in T.index:
            ibb = T.loc[key].values.astype(float); have = ex.loc[key].values if key in ex.index else np.zeros(3)
            rem = np.clip(ibb - have, 0, None)
            p = rem/rem.sum() if rem.sum()>0 else ibb/max(ibb.sum(),1); n_ok += len(ix)
        else:
            p = None; n_miss += len(ix)
        if p is None: continue
        n = len(ix); n3 = int(round(n*p[2])); n2 = int(round(n*p[1]))
        order = ix[np.argsort(-area[ix])]
        c = np.zeros(n, int); c[:n3] = 2; c[n3:n3+n2] = 1
        kat_new[order] = np.where(c==2, 11, np.where(c==1, 6, np.where(area[order]>150, 3, 2)))
    B["kat"] = kat_new; B.loc[target, "h"] = B.loc[target, "kat"]*3.2
    B.drop(columns="key").to_parquet(fp)
    log("kat düzeltme: mahalle eşleşen eklenen bina", n_ok, "| eşleşmeyen", n_miss)
    for il in ["Pendik","Tuzla","Beykoz","Kadıköy","Fatih","Esenyurt","Silivri"]:
        m = (B.ilce==il).values
        log(f"  {il}: medyan h {np.median(B.h.values[m]):.1f} m | 5+ kat pay %{100*np.mean(B.kat.values[m]>=5):.0f}")
    open(mark,"w").write("ok")

# ================= 4. yol ağı
def carriageway(df):
    t = df.other_tags.fillna("")
    hw = df.highway.fillna("").str.replace("_link","",regex=False)
    w = pd.to_numeric(t.str.extract(r'"width"=>"([\d.]+)')[0], errors="coerce")
    lanes = pd.to_numeric(t.str.extract(r'"lanes"=>"(\d+)')[0], errors="coerce")
    oneway = t.str.contains(r'"oneway"=>"(?:yes|1|true|-1)"')
    default = hw.map({"motorway":14,"trunk":12,"primary":11,"secondary":9,"tertiary":7.5,"unclassified":6,"residential":6,"living_street":4.5,"service":4,"road":6}).fillna(6)
    default = np.where(oneway & hw.isin(["residential","unclassified","tertiary"]), default*0.75, default)
    cw = np.where((w>2)&(w<60), w, np.where(lanes.notna(), lanes*3.2, default))
    src = np.where((w>2)&(w<60), "width", np.where(lanes.notna(), "lanes", "sinif"))
    return hw, np.round(cw,1), src

def s4_ag():
    out = os.path.join(D,"ag.pkl")
    if os.path.exists(out): return
    R = gpd.read_file(OSM_IL, layer="roads")
    R = R.explode(index_parts=False).reset_index(drop=True)
    R["hw"], R["cw"], R["cw_src"] = carriageway(R)
    lines = [np.asarray(g.coords) for g in R.geometry]
    keys_ = [np.round(a*10).astype(np.int64) for a in lines]
    cnt = Counter()
    for k in keys_: cnt.update(set(map(tuple,k)))
    node_id={}; rows=[]
    hw=R.hw.values; cw=R.cw.values; cs=R.cw_src.values; nm=R.name.values; oid=R.osm_id.values
    for li,(a,k) in enumerate(zip(lines,keys_)):
        kt = list(map(tuple,k)); start=0
        for i in range(1,len(kt)):
            if cnt[kt[i]]>=2 or i==len(kt)-1:
                seg=a[start:i+1]
                if len(seg)>=2:
                    u=node_id.setdefault(kt[start],len(node_id)); v=node_id.setdefault(kt[i],len(node_id))
                    rows.append((u,v,li,LineString(seg)))
                start=i
    u,v,li,g = zip(*rows)
    E = gpd.GeoDataFrame(dict(u=np.array(u,np.int32), v=np.array(v,np.int32), hw=hw[list(li)], cw=cw[list(li)], cw_src=cs[list(li)], name=nm[list(li)], osm_id=oid[list(li)]), geometry=list(g), crs=CRS)
    E = E[E.length>0.2].reset_index(drop=True); E["uzunluk"]=E.length; E["eid"]=np.arange(len(E))
    xy = np.array(list(node_id.keys()),dtype=float)/10.0
    N = gpd.GeoDataFrame({"nid":np.arange(len(xy))}, geometry=gpd.points_from_xy(xy[:,0],xy[:,1]), crs=CRS)
    A = sp.coo_matrix((np.ones(len(E)),(E.u.values,E.v.values)), shape=(len(N),len(N)))
    nc, lab = cg.connected_components(A, directed=False)
    size = np.bincount(lab); N["comp"]=lab; N["big"]=size[lab]>=MIN_COMP
    E.to_parquet(os.path.join(D,"kenarlar.parquet")); N.to_parquet(os.path.join(D,"dugumler.parquet"))
    pickle.dump(dict(n_edges=len(E), n_nodes=len(N), n_comp=nc), open(out,"wb"))
    log("ağ: kenar", len(E), "düğüm", len(N), "bileşen", nc, "büyük bileşen düğüm payı", round(N.big.mean(),4), "| en büyük 3:", sorted(size)[-3:])

# ================= 5. bina–yol enkaz çiftleri (ilçe ilçe)
def s5_ciftler():
    out = os.path.join(D,"ciftler.parquet")
    if os.path.exists(out): return
    B = gpd.read_parquet(os.path.join(D,"binalar.parquet")); E = gpd.read_parquet(os.path.join(D,"kenarlar.parquet"))
    Eg = E.geometry.values; cwE = E.cw.values.astype(np.float32); Es = E.sindex
    parts=[]
    for il, idx in B.groupby("ilce").indices.items():
        bg_all = B.geometry.values[idx]; h_all = B.h.values[idx].astype(np.float32)
        bi, ei = Es.query(bg_all, predicate="dwithin", distance=50.0)
        bg = bg_all[bi]; eg = Eg[ei]; h=h_all[bi]; cw=cwE[ei]
        d = shapely.distance(bg, eg); gap = np.clip(d - cw/2, 0, None)
        i_full = np.clip(K_FULL*h-gap,0,cw); i_part = np.clip(K_PART*h-gap,0,cw); keep=i_full>0
        bi,ei,bg,eg,h,i_full,i_part = bi[keep],ei[keep],bg[keep],eg[keep],h[keep],i_full[keep],i_part[keep]
        cen = shapely.centroid(bg); s = shapely.line_locate_point(eg,cen); L = shapely.length(eg)
        p0 = shapely.line_interpolate_point(eg, np.clip(s-1,0,L)); p1 = shapely.line_interpolate_point(eg, np.clip(s+1,0,L))
        x0,y0,x1,y1 = shapely.get_x(p0),shapely.get_y(p0),shapely.get_x(p1),shapely.get_y(p1); cx,cy = shapely.get_x(cen),shapely.get_y(cen)
        side = ((x1-x0)*(cy-y0)-(y1-y0)*(cx-x0) >= 0).astype(np.int8)
        half = np.sqrt(shapely.area(bg))/2 + K_FULL*h*0.5
        b0 = np.floor(np.clip(s-half,0,L)/BIN).astype(np.int32); b1 = np.floor(np.maximum(np.clip(s+half,0,L)-1e-6,0)/BIN).astype(np.int32)
        n = b1-b0+1; rep = np.repeat(np.arange(len(bi)), n); off = np.arange(n.sum()) - np.repeat(np.cumsum(n)-n, n)
        parts.append(pd.DataFrame(dict(b=idx[bi[rep]].astype(np.int32), e=ei[rep].astype(np.int32), bin=(b0[rep]+off).astype(np.int32), side=side[rep],
                                       i_full=i_full[rep].astype(np.float32), i_part=i_part[rep].astype(np.float32))))
        log("çiftler", il, len(idx), "bina ->", len(parts[-1]), "satır"); gc.collect()
    P = pd.concat(parts, ignore_index=True); P.to_parquet(out); log("çiftler toplam", len(P))

# ================= 6. senaryo olasılıkları, nüfus, hastaneler
def s6_senaryo():
    out = os.path.join(D,"bina_attr.parquet")
    if os.path.exists(out): return
    B = gpd.read_parquet(os.path.join(D,"binalar.parquet"))
    S = pd.read_csv(os.path.join(EV,"data","raw","ibb_deprem_senaryo.csv"), sep=";", encoding="cp1254")
    T = pd.read_csv(os.path.join(EV,"data","raw","ibb_mahalle_bina.csv"), sep=";", encoding="cp1254")
    T["toplam"]=T[["1980_oncesi","1980-2000_arasi","2000_sonrasi"]].sum(axis=1)
    S = S.merge(T[["mahalle_uavt","toplam"]], left_on="mahalle_koy_uavt", right_on="mahalle_uavt")
    S["p_cok"]=S.cok_agir_hasarli_bina_sayisi/S.toplam; S["p_agir"]=S.agir_hasarli_bina_sayisi/S.toplam
    S["k_ilce"]=S.ilce_adi.map(norm).replace({"EYUP":"EYUPSULTAN"}); S["k_mah"]=S.mahalle_adi.map(norm)
    il = S.groupby("k_ilce").apply(lambda g: pd.Series(dict(pc_i=g.cok_agir_hasarli_bina_sayisi.sum()/g.toplam.sum(), pa_i=g.agir_hasarli_bina_sayisi.sum()/g.toplam.sum())))
    A = pd.DataFrame(dict(k_ilce=B.ilce.map(norm).values, k_mah=B.mah.map(norm).values))
    A = A.merge(S[["k_ilce","k_mah","p_cok","p_agir"]].drop_duplicates(["k_ilce","k_mah"]), on=["k_ilce","k_mah"], how="left")
    A["eslesme"]=np.where(A.p_cok.notna(),"mahalle","ilce")
    A = A.merge(il, left_on="k_ilce", right_index=True, how="left")
    A["p_cok"]=A.p_cok.fillna(A.pc_i); A["p_agir"]=A.p_agir.fillna(A.pa_i)
    MS = pd.read_parquet(os.path.join(D,"mahalle_senaryolar.parquet")); mm = MS.drop_duplicates("key").set_index("key")
    A["key"] = A.k_ilce + "|" + A.k_mah
    for src in ("ibb","dogu","birlesik"):
        ilc = MS.groupby("k_ilce").apply(lambda g: pd.Series(dict(c=(g[f"p_cok_{src}"]*g.toplam).sum()/g.toplam.sum(), a=(g[f"p_agir_{src}"]*g.toplam).sum()/g.toplam.sum())))
        A[f"p_cok_{src}"] = A.key.map(mm[f"p_cok_{src}"]).fillna(A.k_ilce.map(ilc.c)).astype(np.float32)
        A[f"p_agir_{src}"] = A.key.map(mm[f"p_agir_{src}"]).fillna(A.k_ilce.map(ilc.a)).astype(np.float32)
    # nüfus: ilçe nüfusu, kat alanıyla orantılı dağıtılır
    fa = B.area.values*B.kat.fillna(3).values; fa = np.where(B.area.values<40,0,fa)
    pop = np.zeros(len(B))
    for ilce, ix in B.groupby("ilce").indices.items():
        NN = {norm(k):v for k,v in NUFUS.items()}; w = fa[ix]; pop[ix] = w/w.sum()*NN.get(norm(ilce), 0)
        if norm(ilce) not in NN: log("UYARI nüfus yok:", ilce)
    out_df = pd.DataFrame(dict(p_cok=A.p_cok.values.astype(np.float32), p_agir=A.p_agir.values.astype(np.float32), eslesme=A.eslesme.values, nufus=pop.astype(np.float32)))
    for src in ("ibb","dogu","birlesik"): out_df[f"p_cok_{src}"]=A[f"p_cok_{src}"].values; out_df[f"p_agir_{src}"]=A[f"p_agir_{src}"].values
    out_df.to_parquet(out)
    log("senaryo eşleşme:", out_df.eslesme.value_counts().to_dict(), "| ilçe olasılığı eksik:", int(out_df.p_cok.isna().sum()))
    x = pd.read_excel(os.path.join(ROOT,"data","ibb_saglik_tesisleri.xlsx"))
    hk = ["Devlet Hastanesi","Şehir Hastanesi","Eğitim Araştırma Hastanesi","Üniversite Hastanesi","Özel Hastane","Kadın Doğum ve Çocuk Hastanesi"]
    x = x[x["Alt Kategori"].isin(hk)].dropna(subset=["Latitude","Longitude"])
    H = gpd.GeoDataFrame(x.rename(columns={"Sağlık Tesisi Adı":"ad","Alt Kategori":"tur","İlçe Adı":"ilce_h"})[["ad","tur","ilce_h"]], geometry=gpd.points_from_xy(x.Longitude,x.Latitude), crs="EPSG:4326").to_crs(CRS)
    H.to_parquet(os.path.join(D,"hastaneler.parquet")); log("hastane", len(H))

# ================= 7. toplanma alanları (kendi kuralımız)
def s7_toplanma():
    out = os.path.join(D,"toplanma.parquet")
    if os.path.exists(out): return
    B = gpd.read_parquet(os.path.join(D,"binalar.parquet"), columns=["h","geometry"])
    I = gpd.read_file(os.path.join(D,"ilceler.gpkg"))
    A = gpd.read_file(OSM_IL, layer="acik_alan_aday")
    A = A[(A.amenity.fillna("")!="parking") & A.military.isna()].copy()
    A["geometry"] = shapely.make_valid(A.geometry.values); A = A[A.geometry.geom_type.isin(["Polygon","MultiPolygon"])]
    def tur(r):
        if r.leisure in ("park","garden","common","recreation_ground") or r.landuse in ("grass","recreation_ground","village_green","meadow"): return "park/bahçe"
        if r.leisure in ("playground","pitch","sports_centre","stadium","track"): return "spor/oyun"
        if r.amenity in ("school","university","college","kindergarten"): return "okul bahçesi"
        if r.landuse=="cemetery" or r.amenity=="grave_yard": return "mezarlık"
        return "meydan/yaya"
    A["tur"]=A.apply(tur,axis=1)
    j = gpd.sjoin(gpd.GeoDataFrame(geometry=A.representative_point(), crs=CRS), I, how="left", predicate="within"); j=j[~j.index.duplicated()]
    A["ilce"]=j.ilce.values; A=A[A.ilce.notna()]
    pri = {"park/bahçe":0,"meydan/yaya":1,"spor/oyun":2,"okul bahçesi":3,"mezarlık":4}
    res=[]
    for ilce, g in A.groupby("ilce"):
        U = shapely.union_all(g.geometry.values)
        near = B.iloc[np.unique(B.sindex.query(shapely.buffer(U,45), predicate="intersects"))]
        dist = np.clip(near.h.fillna(9).values, 5, 40)
        net = shapely.difference(U, shapely.union_all(shapely.buffer(near.geometry.values, dist, quad_segs=4)))
        p = gpd.GeoDataFrame(geometry=shapely.get_parts(net), crs=CRS); p = p[p.area>=100]
        p = p[~shapely.is_empty(shapely.buffer(p.geometry.values,-5))]; p = p[p.area>=300].reset_index(drop=True)
        rp = shapely.point_on_surface(p.geometry.values); pi, ai = g.sindex.query(rp, predicate="intersects")
        df = pd.DataFrame(dict(p=pi, tur=g.tur.values[ai], ad=g.name.values[ai])); df["o"]=df.tur.map(pri); df=df.sort_values("o").drop_duplicates("p")
        p["tur"]=None; p["ad"]=None; p.loc[df.p.values,"tur"]=df.tur.values; p.loc[df.p.values,"ad"]=df.ad.values; p["ilce"]=ilce
        res.append(p); log("toplanma", ilce, len(p), "alan", round(p.area.sum()/1e4,1), "ha")
    Tt = pd.concat(res, ignore_index=True); Tt["alan"]=Tt.area; Tt["kapasite"]=Tt.alan/M2_KISI; Tt["tid"]=np.arange(len(Tt))
    Tt.to_parquet(out); log("toplanma toplam", len(Tt))

# ================= 8. simülasyon
def nearest_big(Nb, geoms):
    return Nb.nid.values[Nb.sindex.nearest(geoms, return_all=False)[1]]

def s8_sim():
    out = os.path.join(D,"sim.pkl")
    if os.path.exists(out): return
    B = gpd.read_parquet(os.path.join(D,"binalar.parquet"), columns=["ilce","geometry"])
    BA = pd.read_parquet(os.path.join(D,"bina_attr.parquet"))
    E = pd.read_parquet(os.path.join(D,"kenarlar.parquet"), columns=["u","v","hw","cw","uzunluk"])
    N = gpd.read_parquet(os.path.join(D,"dugumler.parquet")); H = gpd.read_parquet(os.path.join(D,"hastaneler.parquet"))
    Tt = gpd.read_parquet(os.path.join(D,"toplanma.parquet"))
    P = pd.read_parquet(os.path.join(D,"ciftler.parquet"))
    n=len(N); nb=len(B); ne=len(E)
    big = N.big.values; emask = big[E.u.values] & big[E.v.values]
    Nb = N[big]
    bnode = nearest_big(Nb, B.centroid); hnode = np.unique(nearest_big(Nb, H.geometry))
    # toplanma alanı giriş düğümleri
    ti, ni = Nb.sindex.query(Tt.geometry, predicate="dwithin", distance=25.0)
    miss = np.setdiff1d(np.arange(len(Tt)), ti)
    if len(miss): ti=np.r_[ti,miss]; ni=np.r_[ni, Nb.sindex.nearest(Tt.geometry.iloc[miss], return_all=False)[1]]
    tn = Nb.nid.values[ni]; order=np.argsort(-Tt.alan.values[ti]); node2area = np.full(n,-1,np.int64)
    for k in order[::-1]: node2area[tn[k]] = ti[k]
    tsrc = np.unique(tn); cap = Tt.kapasite.values
    log("sim hazırlık: bina", nb, "kenar", ne, "hastane düğümü", len(hnode), "toplanma girişi", len(tsrc), "çift", len(P))
    # (kenar,bin,taraf) grupları — sıralı, reduceat için
    k3 = (P.e.values.astype(np.int64)*100000 + P.bin.values)*2 + P.side.values
    order = np.argsort(k3, kind="stable"); k3s = k3[order]
    starts = np.r_[0, np.flatnonzero(np.diff(k3s))+1]; u3 = k3s[starts]
    pb = P.b.values[order]; ifull = P.i_full.values[order]; ipart = P.i_part.values[order]
    del P, k3, order; gc.collect()
    k2 = u3//2; starts2 = np.r_[0, np.flatnonzero(np.diff(k2))+1]; e2 = (k2[starts2]//100000).astype(np.int64); cw2 = E.cw.values[e2]
    grp3to2 = np.repeat(np.arange(len(starts2)), np.diff(np.r_[starts2, len(u3)]))
    tmin = (E.uzunluk.values/1000/E.hw.map(SPEED).fillna(DEFAULT_SPEED).values*60)
    U_, V_ = E.u.values, E.v.values; Lm = E.uzunluk.values
    def graph(w, m):
        return sp.csr_matrix((np.r_[w[m],w[m]], (np.r_[U_[m],V_[m]], np.r_[V_[m],U_[m]])), shape=(n,n))
    def hosp(blk):
        return cg.dijkstra(graph(tmin, emask & ~blk), directed=False, indices=hnode, min_only=True)[bnode]
    def walk(blk):
        d,_,s = cg.dijkstra(graph(Lm, emask & ~blk), directed=False, indices=tsrc, min_only=True, return_predecessors=True)
        db = d[bnode]; sb = s[bnode]; a = np.where(sb>=0, node2area[np.clip(sb,0,None)], -1); return db, a
    base_h = hosp(np.zeros(ne,bool)); base_w, base_a = walk(np.zeros(ne,bool))
    log("baz: hastaneye erişimsiz bina (bileşen sorunu)", int((~np.isfinite(base_h)).sum()), "| toplanmaya erişimsiz", int((~np.isfinite(base_w)).sum()))
    pop = BA.nufus.values.astype(np.float64)
    R = dict(base_h=base_h.astype(np.float32), base_w=base_w.astype(np.float32), base_a=base_a.astype(np.int32))
    for name, src, mult in SCEN:
        rng = np.random.default_rng(7)
        pc = np.clip(BA[f"p_cok_{src}"].values*mult,0,1); pa = np.clip(BA[f"p_agir_{src}"].values*mult,0,1)
        iso=np.zeros(nb,np.float32); cut=np.zeros(nb,np.float32); far=np.zeros(nb,np.float32); dts=np.zeros(nb,np.float32)
        blk_c=np.zeros(ne,np.float32); blkw_c=np.zeros(ne,np.float32); col=np.zeros(nb,np.float32); load=np.zeros(len(Tt)); stats=[]
        for r in range(N_RUNS):
            u = rng.random(nb); st = np.where(u<pc,2,np.where(u<pc+pa,1,0)).astype(np.int8)
            s = st[pb]; intr = np.where(s==2, ifull, np.where(s==1, ipart, np.float32(0)))
            m3 = np.maximum.reduceat(intr, starts)
            tot2 = np.bincount(grp3to2, weights=m3, minlength=len(starts2))
            clear = cw2 - tot2
            blk = np.zeros(ne,bool); blk[e2[clear<MIN_CLEAR]] = True
            blkw = np.zeros(ne,bool); blkw[e2[clear<WALK_CLEAR]] = True
            d = hosp(blk); dw, aw = walk(blkw)
            alive = st<2; fin = np.isfinite(base_h)
            i_ = ~np.isfinite(d) & fin; iso += i_; dts += np.where(np.isfinite(d)&fin, d-base_h, 0)
            c_ = ~np.isfinite(dw) & np.isfinite(base_w) & alive; cut += c_; far += (np.isfinite(dw) & (dw>YAKIN_M) & alive)
            ok = (aw>=0) & alive & np.isfinite(dw); load += np.bincount(aw[ok], weights=pop[ok], minlength=len(Tt))
            blk_c += blk; blkw_c += blkw; col += (st==2)
            stats.append((r, int(blk.sum()), float(Lm[blk].sum()/1000), int(i_.sum()), float(pop[i_].sum()), float(pop[c_].sum())))
            if r % 10 == 0: log(name, "tur", r, "kapalı km", round(stats[-1][2]), "izole bina", stats[-1][3])
        R[name] = dict(p_iso=iso/N_RUNS, p_cut=cut/N_RUNS, p_far=far/N_RUNS, dt=dts/N_RUNS, p_blk=blk_c/N_RUNS, p_blkw=blkw_c/N_RUNS, p_col=col/N_RUNS, load=load/N_RUNS,
                       stats=pd.DataFrame(stats, columns=["run","kapali_kenar","kapali_km","izole_bina","izole_nufus","toplanma_kopuk_nufus"]))
        st_ = R[name]["stats"]; log(f"== {name}: izole nüfus ort {st_.izole_nufus.mean():,.0f} | toplanmadan kopuk nüfus {st_.toplanma_kopuk_nufus.mean():,.0f} | kapalı yol {st_.kapali_km.mean():.0f} km")
    pickle.dump(R, open(out,"wb")); log("sim kaydedildi")


# ================= 9. çıktılar
def s9_cikti():
    out = os.path.join(D,"sonuc_il.gpkg")
    R = pickle.load(open(os.path.join(D,"sim.pkl"),"rb"))
    B = gpd.read_parquet(os.path.join(D,"binalar.parquet")); BA = pd.read_parquet(os.path.join(D,"bina_attr.parquet"))
    Tt = gpd.read_parquet(os.path.join(D,"toplanma.parquet")); H = gpd.read_parquet(os.path.join(D,"hastaneler.parquet"))
    pop = BA.nufus.values.astype(float)
    B["nufus"]=pop.round(1); B["p_cok"]=BA.p_cok.values; B["p_agir"]=BA.p_agir.values
    B["hastane_dk_baz"]=np.round(R["base_h"],1); B["toplanma_m_baz"]=np.round(R["base_w"],0); B["toplanma_id"]=R["base_a"]
    for k,*_ in SCEN:
        B[f"izole_{k}"]=(100*R[k]["p_iso"]).round(1); B[f"toplanma_kopuk_{k}"]=(100*R[k]["p_cut"]).round(1); B[f"cokme_{k}"]=(100*R[k]["p_col"]).round(1)
        Tt[f"yuk_{k}"]=R[k]["load"].round(0); Tt[f"doluluk_{k}"]=(100*R[k]["load"]/Tt.kapasite.values).round(0)
    # taşan toplanma alanına bağlı nüfus
    for k,*_ in SCEN:
        over = (R[k]["load"] > Tt.kapasite.values); a = R["base_a"]
        B[f"toplanma_tasan_{k}"] = np.where(a>=0, over[np.clip(a,0,None)], False)
    def wmean(g, c): return (g[c]*g.nufus).sum()/max(g.nufus.sum(),1)
    # ilçe özeti
    E = pd.read_parquet(os.path.join(D,"kenarlar.parquet"), columns=["uzunluk"]); Eg = gpd.read_parquet(os.path.join(D,"kenarlar.parquet"))
    I = gpd.read_file(os.path.join(D,"ilceler.gpkg"))
    ji = gpd.sjoin(gpd.GeoDataFrame(geometry=Eg.representative_point(), crs=CRS), I, how="left", predicate="within"); ji=ji[~ji.index.duplicated()]
    Eg["ilce"]=ji.ilce.values
    rows=[]
    for il, g in B.groupby("ilce"):
        r = dict(ilce=il, bina=len(g), nufus=round(g.nufus.sum()))
        em = (Eg.ilce==il).values
        for k,*_ in SCEN:
            r[f"hastane_kopuk_%_{k}"]=round(wmean(g,f"izole_{k}"),1); r[f"toplanma_kopuk_%_{k}"]=round(wmean(g,f"toplanma_kopuk_{k}"),1)
            r[f"kapali_km_{k}"]=round(float((R[k]["p_blk"][em]*Eg.uzunluk.values[em]).sum()/1000),1)
            r[f"tasan_alan_nufus_%_{k}"]=round(100*g.nufus[g[f"toplanma_tasan_{k}"]].sum()/max(g.nufus.sum(),1),1)
        t = Tt[Tt.ilce==il]; r["toplanma_alan_sayisi"]=len(t); r["toplanma_kapasite"]=round(t.kapasite.sum()); r["kapasite_nufus_orani"]=round(t.kapasite.sum()/max(g.nufus.sum(),1),2)
        rows.append(r)
    IL = pd.DataFrame(rows).sort_values("hastane_kopuk_%_ibb", ascending=False)
    IL.to_csv(os.path.join(D,"ilce_ozet.csv"), index=False, encoding="utf-8-sig"); log("ilçe özeti yazıldı")
    I.merge(IL, on="ilce").to_file(out, layer="ilceler", driver="GPKG")
    # mahalle özeti — OSM mahalle poligonlarına
    Mh = gpd.read_file(OSM_IL, layer="mahalle"); Mh = Mh[Mh.admin_level=="8"][["name","geometry"]].reset_index(drop=True); Mh["mid"]=np.arange(len(Mh))
    jm = gpd.sjoin(gpd.GeoDataFrame(geometry=B.centroid, crs=CRS), Mh, how="left", predicate="within"); jm=jm[~jm.index.duplicated()]
    B["mid"]=jm.mid.values
    agg=[]
    for mid, g in B[B.mid.notna()].groupby("mid"):
        r=dict(mid=int(mid), ilce=g.ilce.mode().iat[0], bina=len(g), nufus=round(g.nufus.sum()))
        for k,*_ in SCEN:
            r[f"hastane_kopuk_{k}"]=round(wmean(g,f"izole_{k}"),1); r[f"toplanma_kopuk_{k}"]=round(wmean(g,f"toplanma_kopuk_{k}"),1)
            r[f"tasan_{k}"]=round(100*g.nufus[g[f"toplanma_tasan_{k}"]].sum()/max(g.nufus.sum(),1),1)
        agg.append(r)
    MM = Mh.merge(pd.DataFrame(agg), on="mid"); MM = MM[MM.bina>=30]
    MM.to_file(out, layer="mahalleler", driver="GPKG")
    MM.drop(columns="geometry").sort_values("hastane_kopuk_ibb", ascending=False).to_csv(os.path.join(D,"mahalle_ozet.csv"), index=False, encoding="utf-8-sig")
    log("mahalle özeti", len(MM))
    Eo = Eg[["eid","name","hw","cw","uzunluk","ilce","geometry"]].copy()
    for k,*_ in SCEN: Eo[f"kapanma_{k}"]=(100*R[k]["p_blk"]).round(1)
    Eo.to_file(out, layer="yol_kapanma", driver="GPKG"); log("yollar yazıldı")
    Tt.to_file(out, layer="toplanma_alanlari", driver="GPKG"); H.to_file(out, layer="hastaneler", driver="GPKG")
    keep = ["bid","ilce","mah","kat","h","kaynak","nufus","hastane_dk_baz","toplanma_m_baz"]+[f"izole_{k}" for k,*_ in SCEN]+[f"toplanma_kopuk_{k}" for k,*_ in SCEN]+["geometry"]
    B[keep].to_file(os.path.join(D,"binalar_sonuc.gpkg"), layer="binalar", driver="GPKG"); log("binalar yazıldı")
    tot = pop.sum()
    for k,*_ in SCEN:
        st = R[k]["stats"]
        log(f"ÖZET {k}: hastaneden kopan nüfus %{100*st.izole_nufus.mean()/tot:.1f} ({st.izole_nufus.mean():,.0f}) | toplanmadan kopuk %{100*st.toplanma_kopuk_nufus.mean()/tot:.1f} | kapalı yol {st.kapali_km.mean():.0f} km | taşan alana bağlı nüfus %{100*pop[B[f'toplanma_tasan_{k}'].values].sum()/tot:.1f}")
    log("toplam nüfus", round(tot), "| toplanma kapasitesi", round(Tt.kapasite.sum()))

STEPS = [s1_binalar, s0_osm, s2_overture, s3_tamamla, s3b_kat, s4_ag, s5_ciftler, s6_senaryo, s7_toplanma, s8_sim, s9_cikti]
if __name__ == "__main__":
    only = sys.argv[1:]
    try:
        for f in STEPS:
            if only and f.__name__ not in only: continue
            log(">>>", f.__name__); f(); gc.collect()
        log("BİTTİ")
    except Exception as e:
        import traceback; log("HATA", repr(e)); log(traceback.format_exc())
