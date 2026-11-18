import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np, os
OUT=os.path.dirname(os.path.abspath(__file__))+"/"
X =np.array([[1,-16,2,6],[-2,8,-1,-9]])
W1=np.array([[2,1,-2,1],[1,-1,-1,-1]]); W2=np.array([[1,4,2,-1],[-1,-2,1,1]])
xmax=np.abs(X).max(0); wt=np.maximum(np.abs(W1).max(0),np.abs(W2).max(0)); s=np.sqrt(xmax/wt)
Xh=X/s; W1h=W1*s; W2h=W2*s
INK="#222"; MUT="#6b6a64"; GR="#8a8a8a"; ACC="#b03a2e"; HL=1; cs=0.42; W=4*cs; H=2*cs
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
def line(ax,pts,c,lw=1.0): ax.plot([p[0] for p in pts],[p[1] for p in pts],color=c,lw=lw,solid_capstyle="butt")
def arr(ax,a,b,c,lw=1.0): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=8,lw=lw,color=c,shrinkA=0,shrinkB=0))
def bracket(ax,x,y0,y1,c,side):   # side=+1 opens left (right bracket), -1 opens right
    ax.plot([x-side*0.07,x,x,x-side*0.07],[y0,y0,y1,y1],color=c,lw=1)
fig,ax=plt.subplots(figsize=(7.4,3.1))
# grid
x0=0.0; x1=x0+W+0.55; mx=x1+W+0.2; x2=mx+0.38; sx=x2+W+0.2; x3=sx+0.5
yXt=3.0; yX=yXt-H/2                          # x row centre
yW1t,yW2t=1.72,0.42; yW=(yW1t+(yW2t-H))/2    # W stack centre
yS=(yX+yW)/2
# column 0: inputs
mat(ax,X,x0,yXt,16);  T(ax,x0+W/2,yXt+0.06,"$x$",ha="center",va="bottom")
mat(ax,W1,x0,yW1t,4); T(ax,x0+W/2,yW1t+0.06,"$W^{(1)}$",ha="center",va="bottom")
mat(ax,W2,x0,yW2t,4); T(ax,x0+W/2,yW2t+0.06,"$W^{(2)}$",ha="center",va="bottom")
bracket(ax,x0+W+0.1,yW1t,yW2t-H,ACC,+1); T(ax,x0-0.08,yW,"$\\mathcal{C}(g)$",c=ACC,va="center",ha="right",fs=8.5)
# column 1: tied statistics (one row each, aligned with its input)
mat(ax,xmax[None],x1,yX+cs/2,16,dashed=True); T(ax,x1+W/2,yX+cs/2+0.06,"$\\max|x_c|$",fs=8,c=MUT,ha="center",va="bottom")
mat(ax,wt[None],x1,yW+cs/2,4,dashed=True);    T(ax,x1+W/2,yW+cs/2+0.06,"$\\mathrm{max}_{j}\\max|W^{(j)}_{:,c}|$",fs=8,c=MUT,ha="center",va="bottom")
arr(ax,(x0+W+0.05,yX),(x1-0.06,yX),GR); arr(ax,(x0+W+0.12,yW),(x1-0.06,yW),GR)
# column 2: s (merge)
line(ax,[(x1+W+0.04,yX),(mx,yX),(mx,yW),(x1+W+0.04,yW)],GR); arr(ax,(mx,yS),(x2-0.06,yS),GR)
mat(ax,s[None],x2,yS+cs/2,4,dashed=True); T(ax,x2+W/2,yS+cs/2+0.06,"$s$",fs=10,c=ACC,ha="center",va="bottom")
T(ax,x2+W/2,yS-cs/2-0.08,"$s_c(\\alpha)$",fs=8,c=MUT,ha="center",va="top")
# column 3: outputs (split)
line(ax,[(x2+W+0.04,yS),(sx,yS)],ACC,1.2); line(ax,[(sx,yX),(sx,yW)],ACC,1.2)
arr(ax,(sx,yX),(x3-0.06,yX),ACC,1.2); T(ax,(sx+x3)/2,yX+0.05,"$\\div s$",c=ACC,fs=9,ha="center",va="bottom")
arr(ax,(sx,yW),(x3-0.2,yW),ACC,1.2); T(ax,sx-0.06,yW-0.08,"$\\mathrm{diag}(s)$",c=ACC,fs=8.5,ha="right",va="top")
bracket(ax,x3-0.1,yW1t,yW2t-H,ACC,-1)
mat(ax,Xh,x3,yXt,16);  T(ax,x3+W/2,yXt+0.06,"$x/s$",ha="center",va="bottom")
mat(ax,W1h,x3,yW1t,8); T(ax,x3+W/2,yW1t+0.06,"$W^{(1)}\\mathrm{diag}(s)$",ha="center",va="bottom")
mat(ax,W2h,x3,yW2t,8); T(ax,x3+W/2,yW2t+0.06,"$W^{(2)}\\mathrm{diag}(s)$",ha="center",va="bottom")
# titles
T(ax,(x0+x1+W)/2,3.5,"Tied statistics",fs=10,weight="bold",ha="center")
T(ax,(x2+x3+W)/2,3.5,"TSM (shared $s$)",fs=10,weight="bold",ha="center")
ax.set_xlim(-0.75,x3+W+0.1); ax.set_ylim(-0.45,3.75); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
for suf,kw in (("",dict(dpi=220)),("_transparent",dict(dpi=300,transparent=True))):
    fig.savefig(OUT+f"tsm_numeric_v3{suf}.png",bbox_inches="tight",**kw)
fig.savefig(OUT+"tsm_numeric_v3.pdf",bbox_inches="tight"); print("ok")
