import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
OUT="/tmp/claude-1003/-home-taeho/5692497e-a15d-476f-95db-3ec3f5583e11/scratchpad/fig_pc/"
X =np.array([[1,-16,2,6],[-2,8,-1,-9]])
W1=np.array([[2,1,-2,1],[1,-1,-1,-1]]); W2=np.array([[1,4,2,-1],[-1,-2,1,1]])
xmax=np.abs(X).max(0); wt=np.maximum(np.abs(W1).max(0),np.abs(W2).max(0)); s=np.sqrt(xmax/wt)
Xh=X/s; W1h=W1*s; W2h=W2*s
INK="#222"; MUT="#6b6a64"; ACC="#b03a2e"; HL=1; cs=0.42; W=4*cs; H=2*cs
def col(v,vmax):
    t=min(abs(v)/vmax,1.0); c0=np.array([0.95,0.94,0.91]); c1=np.array([0.96,0.80,0.45]); c2=np.array([0.80,0.36,0.45])
    return tuple(c0+(c1-c0)*t/0.5) if t<0.5 else tuple(c1+(c2-c1)*(t-0.5)/0.5)
fmt=lambda v: f"{int(v)}" if float(v).is_integer() else f"{v:g}"
def mat(ax,M,x,top,vmax,dashed=False):
    r,c=M.shape
    for i in range(r):
        for j in range(c):
            ax.add_patch(Rectangle((x+j*cs,top-(i+1)*cs),cs,cs,fc=col(M[i,j],vmax),ec="white",lw=1.2))
            ax.text(x+j*cs+cs/2,top-(i+0.5)*cs,fmt(M[i,j]),ha="center",va="center",fontsize=8.5,color=INK)
    ax.add_patch(Rectangle((x,top-r*cs),c*cs,r*cs,fill=False,ec="#777",lw=0.9,ls=(0,(3,2)) if dashed else "-"))
    ax.add_patch(Rectangle((x+HL*cs,top-r*cs),cs,r*cs,fill=False,ec=ACC,lw=1.6,zorder=5))
def T(ax,x,y,t,fs=9,c=INK,**k): ax.text(x,y,t,fontsize=fs,color=c,**k)
def arr(ax,a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=7,lw=0.8,color="#888"))
fig,ax=plt.subplots(figsize=(7.6,3.6))
TOP1,TOP2,STAT,SROW=3.0,1.72,0.25,-0.72
xL,xW=0.0,2.3                              # left panel columns
# --- left: tied statistics ---
T(ax,(xL+xW+W)/2,3.55,"Tied statistics",fs=10,weight="bold",ha="center")
mat(ax,X,xL,TOP1,16);   T(ax,xL+W/2,TOP1+0.06,"$x$",ha="center",va="bottom")
mat(ax,W1,xW,TOP1,4);   T(ax,xW+W/2,TOP1+0.06,"$W^{(1)}$",ha="center",va="bottom")
mat(ax,W2,xW,TOP2,4);   T(ax,xW+W/2,TOP2+0.06,"$W^{(2)}$",ha="center",va="bottom")
bx=xW+W+0.1; ax.plot([bx,bx+0.07,bx+0.07,bx],[TOP1,TOP1,TOP2-H,TOP2-H],color=ACC,lw=1)
T(ax,bx+0.13,(TOP1+TOP2-H)/2,"$\\mathcal{C}(g)$",c=ACC,va="center")
mat(ax,xmax[None],xL,STAT,16,dashed=True); T(ax,xL+W/2,STAT+0.06,"$\\max|x_c|$",fs=8,c=MUT,ha="center",va="bottom")
mat(ax,wt[None],xW,STAT,4,dashed=True);    T(ax,xW+W/2,STAT+0.06,"$\\mathrm{max}_{j}\\max|W^{(j)}_{:,c}|$",fs=8,c=MUT,ha="center",va="bottom")
arr(ax,(xL+W/2,TOP1-H-0.05),(xL+W/2,STAT+0.32)); arr(ax,(xW+W/2,TOP2-H-0.05),(xW+W/2,STAT+0.32))
sx=bx+0.75                                   # s sits between the two panels
mat(ax,s[None],sx,SROW,4,dashed=True)
T(ax,sx+W/2,SROW-cs-0.08,"$s_c=\\sqrt{\\max|x_c|\\,/\\,\\mathrm{max}_j\\max|W^{(j)}_{:,c}|}$",fs=8,c=MUT,ha="center",va="top")
T(ax,sx+W/2,SROW+0.06,"$s$",fs=10,c=ACC,ha="center",va="bottom")
arr(ax,(xL+W/2,STAT-cs-0.05),(sx-0.05,SROW-cs/2-0.05)); arr(ax,(xW+W/2,STAT-cs-0.05),(sx-0.05,SROW-cs/2+0.05))
dv=sx+W-0.1
# --- right: TSM (shared s), rows aligned with the left ---
xR,xRW=dv+0.4,dv+0.4+2.3
T(ax,(xR+xRW+W)/2,3.55,"TSM (shared $s$)",fs=10,weight="bold",ha="center")
mat(ax,Xh,xR,TOP1,16);  T(ax,xR+W/2,TOP1+0.06,"$x/s$",ha="center",va="bottom")
T(ax,xR+W/2,TOP1-H-0.1,"once, at the producer",fs=7.5,c=MUT,ha="center",va="top")
mat(ax,W1h,xRW,TOP1,8); T(ax,xRW+W/2,TOP1+0.06,"$W^{(1)}\\mathrm{diag}(s)$",ha="center",va="bottom")
mat(ax,W2h,xRW,TOP2,8);
def carr(a,b,rad): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,lw=1.2,color=ACC,connectionstyle=f"arc3,rad={rad}"))
carr((sx+W*0.6,SROW+0.32),(xR+W/2,TOP1-H-0.38),-0.25); T(ax,(sx+xR+W)/2+0.05,(SROW+TOP1-H)/2+0.05,"$\\div s$",c=ACC,fs=9,ha="right")
bxr=xRW-0.12; ax.plot([bxr,bxr-0.07,bxr-0.07,bxr],[TOP1,TOP1,TOP2-H,TOP2-H],color=ACC,lw=1)
carr((sx+W+0.05,SROW-cs/2),(bxr-0.1,(TOP1+TOP2-H)/2-0.35),0.3); T(ax,bxr-0.35,SROW+0.1,"$\\mathrm{diag}(s)$",c=ACC,fs=9,ha="right")
T(ax,xRW+W/2,TOP2+0.06,"$W^{(2)}\\mathrm{diag}(s)$",ha="center",va="bottom")
ax.set_xlim(-0.3,xRW+W+0.2); ax.set_ylim(-1.55,3.8); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
for suf,kw in (("",dict(dpi=220)),("_transparent",dict(dpi=300,transparent=True))):
    fig.savefig(OUT+f"tsm_numeric_v2{suf}.png",bbox_inches="tight",**kw)
fig.savefig(OUT+"tsm_numeric_v2.pdf",bbox_inches="tight"); print("ok")
