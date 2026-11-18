import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
OUT="/tmp/claude-1003/-home-taeho/5692497e-a15d-476f-95db-3ec3f5583e11/scratchpad/fig_pc/"
X =np.array([[1,-16,2,6],[-2,8,-1,-9]])
W1=np.array([[2,1,-2,1],[1,-1,-1,-1]]); W2=np.array([[1,4,2,-1],[-1,-2,1,1]])
xmax=np.abs(X).max(0); w1=np.abs(W1).max(0); w2=np.abs(W2).max(0); wt=np.maximum(w1,w2)
s=np.sqrt(xmax/wt)                        # alpha = 0.5
Xh=(X/s).round(2); W1h=W1*s; W2h=W2*s
INK="#222"; MUT="#6b6a64"; ACC="#b03a2e"
def col(v,vmax=16):
    t=min(abs(v)/vmax,1.0)
    c0=np.array([0.94,0.94,0.91]); c1=np.array([0.96,0.80,0.45]); c2=np.array([0.80,0.36,0.45])
    return tuple(c0+(c1-c0)*t/0.5) if t<0.5 else tuple(c1+(c2-c1)*(t-0.5)/0.5)
def fmt(v): return f"{int(v)}" if float(v).is_integer() else f"{v:g}"
def mat(ax,M,x,y,cs=0.42,dashed=False,hl=None,vmax=16):
    r,c=M.shape
    for i in range(r):
        for j in range(c):
            ax.add_patch(Rectangle((x+j*cs,y-(i+1)*cs),cs,cs,fc=col(M[i,j],vmax),ec="white",lw=1.2))
            ax.text(x+j*cs+cs/2,y-(i+0.5)*cs,fmt(M[i,j]),ha="center",va="center",fontsize=8,color=INK)
    ax.add_patch(Rectangle((x,y-r*cs),c*cs,r*cs,fill=False,ec="#777",lw=0.9,ls=(0,(3,2)) if dashed else "-"))
    if hl is not None: ax.add_patch(Rectangle((x+hl*cs,y-r*cs),cs,r*cs,fill=False,ec=ACC,lw=1.5,zorder=5))
def lab(ax,x,y,t,**k): ax.text(x,y,t,fontsize=k.pop("fs",8.5),color=k.pop("c",INK),**k)
fig,ax=plt.subplots(figsize=(7.4,3.3)); cs=0.42; W=4*cs
# ---------- left: tied statistics ----------
lab(ax,0,3.35,"Tied statistics",fs=9.5,weight="bold")
mat(ax,X,0,3.0,hl=1); lab(ax,W/2,3.06,"$x$",ha="center",va="bottom")
mat(ax,xmax[None],0,2.05,dashed=True,hl=1); lab(ax,-0.08,1.84,"$\\max|x_c|$",ha="right",va="center",fs=7.5,c=MUT)
x2=W+0.75
mat(ax,W1,x2,3.0,vmax=4,hl=1); lab(ax,x2+W/2,3.06,"$W^{(1)}$",ha="center",va="bottom")
mat(ax,W2,x2,1.8,vmax=4,hl=1); lab(ax,x2+W/2,1.86,"$W^{(2)}$",ha="center",va="bottom")
lab(ax,x2+W+0.08,2.15,"$\\mathcal{C}(g)$",va="center",c=ACC)
mat(ax,wt[None],x2,0.75,dashed=True,vmax=4,hl=1); lab(ax,x2-0.08,0.54,"$\\mathrm{max}_{j\\in\\mathcal{C}(g)}\\max|W^{(j)}_{:,c}|$",ha="right",va="center",fs=7.5,c=MUT)
mat(ax,s[None],(x2+0)/2-0.6,-0.05,dashed=True,vmax=4,hl=1)
lab(ax,(x2)/2-0.6+W/2,-0.53,"$s_c=\\sqrt{\\max|x_c|\\,/\\,\\mathrm{max}_{j}\\max|W^{(j)}_{:,c}|}$",ha="center",va="top",fs=7.5,c=MUT)
# ---------- divider ----------
dvx=x2+W+0.85; ax.plot([dvx,dvx],[-0.8,3.45],ls=(0,(3,3)),color="#999",lw=0.9)
# ---------- right: after TSM ----------
x3=dvx+0.35
lab(ax,x3,3.35,"TSM (shared $s$)",fs=9.5,weight="bold")
mat(ax,Xh,x3,3.0,hl=1); lab(ax,x3+W/2,3.06,"$x/s$",ha="center",va="bottom")
lab(ax,x3+W/2,2.05,"once, in producer's requant",ha="center",va="top",fs=7,c=MUT)
x4=x3+W+0.55
mat(ax,W1h,x4,3.0,vmax=8,hl=1); lab(ax,x4+W/2,3.06,"$W^{(1)}\\mathrm{diag}(s)$",ha="center",va="bottom")
mat(ax,W2h,x4,1.8,vmax=8,hl=1); lab(ax,x4+W/2,1.86,"$W^{(2)}\\mathrm{diag}(s)$",ha="center",va="bottom")
ax.set_xlim(-2.2,x4+W+0.3); ax.set_ylim(-0.95,3.6); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
for suf,kw in (("",dict(dpi=220)),("_transparent",dict(dpi=300,transparent=True))):
    fig.savefig(OUT+f"tsm_numeric{suf}.png",bbox_inches="tight",**kw)
fig.savefig(OUT+"tsm_numeric.pdf",bbox_inches="tight")
print("xmax",xmax,"w1",w1,"w2",w2,"tied",wt,"s",s)
