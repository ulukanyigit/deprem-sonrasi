
# senaryo_beta.py — β'yı İBB verisinden kalibre et, doğu senaryolarını üret
import os, sys, numpy as np, pandas as pd
from scipy.stats import norm as N
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(__file__))
import senaryo_model as sm
D = sm.D; log = sm.log
S = pd.read_parquet(os.path.join(D,"mahalle_senaryo_girdi.parquet"))
IBB_IZ = sm.iz(27.95,28.85); r = sm.rjb(S, IBB_IZ); lnA = sm.ln_pga(7.5, r)
W = S[["1980_oncesi","1980-2000_arasi","2000_sonrasi"]].values / S.toplam.values[:,None]
H = S[["5-9 kat_arasi","9-19 kat_arasi"]].values.sum(1) / S.toplam.values
y = S.p_ag.values; w = S.toplam.values

log("=== β kalibrasyonu ===")
# 1) probit regresyonu (hızlı, kapalı form)
m = (y>0.002)&(y<0.6)
X = np.c_[np.ones(m.sum()), lnA[m], W[m,0], W[m,1], H[m]]
sw = np.sqrt(w[m])
coef, *_ = np.linalg.lstsq(X*sw[:,None], N.ppf(y[m])*sw, rcond=None)
beta_probit = 1/coef[1]
log(f"probit eğimi 1/β = {coef[1]:.3f} → β ≈ {beta_probit:.2f}")
# 2) β serbest kısıtlı sınıf modeli (12 başlangıç)
def pred(th):
    b = np.exp(th[4]); return (W * N.cdf((lnA[:,None]-(th[:3]+th[3]*H[:,None]))/b)).sum(1)
f = lambda th: np.sum(w*(pred(th)-y)**2)
rng = np.random.default_rng(1); best=None
for i in range(12):
    x0 = np.r_[np.sort(lnA.mean()+rng.uniform(-0.5,3,3)), rng.uniform(-1,1), np.log(rng.uniform(0.4,1.5))]
    rr = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-5, fatol=1e-12))
    if best is None or rr.fun < best.fun: best=rr
p = pred(best.x); r2 = 1-np.sum(w*(p-y)**2)/np.sum(w*(y-np.average(y,weights=w))**2)
beta_fit = float(np.exp(best.x[4]))
log(f"sınıf modeli, β serbest: β={beta_fit:.2f}  R²={r2:.3f}")
S["rjb"]=r
tab = S.groupby(pd.cut(S.rjb,[0,10,15,20,30,50,100]), observed=True).apply(lambda g: pd.Series(dict(n=len(g), ag_ort=round(100*np.average(g.p_ag,weights=g.toplam),1), ag_p90=round(100*g.p_ag.quantile(.9),1), ag_max=round(100*g.p_ag.max(),1))))
log("İBB: faya uzaklığa göre ağır+çok ağır %\n"+tab.to_string())

def senaryolar(beta):
    ref = sm.ln_pga(7.5, r); EPS=1e-4
    z_ag = N.ppf(np.clip(S.p_ag.values,EPS,1-EPS)); z_cok = N.ppf(np.clip(S.p_cok.values,EPS,1-EPS))
    out = S[["ilce_adi","mahalle_adi","mahalle_koy_uavt","k_ilce","key","toplam","x","y","konum"]].copy()
    out["p_cok_ibb"]=S.p_cok.values; out["p_agir_ibb"]=(S.p_ag-S.p_cok).values
    for k,(line,M) in {"dogu":(sm.iz(28.85,29.40),7.1), "birlesik":(sm.iz(28.25,29.40),7.4)}.items():
        d = (sm.ln_pga(M, sm.rjb(S,line)) - ref)/beta
        pag = np.where(S.p_ag.values==0, 0, N.cdf(z_ag+d)); pc = np.where(S.p_cok.values==0, 0, np.minimum(N.cdf(z_cok+d), pag))
        out[f"p_cok_{k}"]=pc; out[f"p_agir_{k}"]=pag-pc; out[f"dlnpga_{k}"]=(d*beta).round(3)
    return out
def ilce(out):
    return out.groupby("ilce_adi").apply(lambda g: pd.Series({k: round(100*((g[f"p_cok_{k}"]+g[f"p_agir_{k}"])*g.toplam).sum()/g.toplam.sum(),1) for k in ["ibb","dogu","birlesik"]}))
odak = ["ADALAR","TUZLA","PENDİK","KARTAL","MALTEPE","KADIKÖY","ÜSKÜDAR","ATAŞEHİR","BAKIRKÖY","FATİH","AVCILAR","BEYLİKDÜZÜ"]
rows=[]
for b in sorted({0.6, 0.8, 1.0, round(beta_probit,2), round(beta_fit,2)}):
    t = ilce(senaryolar(b)); t["beta"]=b; rows.append(t.reset_index())
R = pd.concat(rows); R.to_csv(os.path.join(D,"senaryo_beta_duyarlilik.csv"), index=False, encoding="utf-8-sig")
piv = R[R.ilce_adi.isin(odak)].pivot_table(index="ilce_adi", columns="beta", values="dogu")
piv.insert(0,"ibb", R[R.beta==R.beta.iloc[0]].set_index("ilce_adi").loc[piv.index,"ibb"])
log("Doğu (Adalar Mw7.1) ağır+çok ağır % — β duyarlılığı\n"+piv.round(1).to_string())
open(os.path.join(D,"beta.txt"),"w").write(f"{beta_probit:.4f},{beta_fit:.4f},{r2:.4f}")
log("BİTTİ")
