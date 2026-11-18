import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
OUT="/tmp/claude-1003/-home-taeho/5692497e-a15d-476f-95db-3ec3f5583e11/scratchpad/fig_pc/"
CH=["#2a78d6","#5fa8e8","#9cc9f0","#f2a65a","#d9603b"]; INK="#222"; MUT="#6b6a64"; ACC="#b03a2e"; HL=1
DX,DY=0.6,0.36   # depth direction (oblique)
def slab(ax,x,y,w,h,d,c,ec="white",lw=0.6,z=2):
    ax.add_patch(Polygon([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],fc=c,ec=ec,lw=lw,zorder=z))                 # front
    ax.add_patch(Polygon([[x,y+h],[x+w,y+h],[x+w+d*DX,y+h+d*DY],[x+d*DX,y+h+d*DY]],fc=c,ec=ec,lw=lw,alpha=0.8,zorder=z))  # top
    ax.add_patch(Polygon([[x+w,y],[x+w+d*DX,y+d*DY],[x+w+d*DX,y+h+d*DY],[x+w,y+h]],fc=c,ec=ec,lw=lw,alpha=0.6,zorder=z))  # side
def tensor(ax,x,y,cw=0.09,h=0.62,d=0.42):
    n=len(CH); w=n*cw
    for i,c in enumerate(CH):   # front + top stripes (no per-slice side faces)
        x0=x+i*cw
        ax.add_patch(Polygon([[x0,y],[x0+cw,y],[x0+cw,y+h],[x0,y+h]],fc=c,ec="white",lw=0.6,zorder=2))
        ax.add_patch(Polygon([[x0,y+h],[x0+cw,y+h],[x0+cw+d*DX,y+h+d*DY],[x0+d*DX,y+h+d*DY]],fc=c,ec="white",lw=0.6,alpha=0.8,zorder=2))
    ax.add_patch(Polygon([[x+w,y],[x+w+d*DX,y+d*DY],[x+w+d*DX,y+h+d*DY],[x+w,y+h]],fc=CH[-1],ec="white",lw=0.6,alpha=0.6,zorder=2))
    gx=x+HL*cw; e=0.07   # group-g slice pops out in place: slightly larger 3D slab
    X,Y,W,H,D=gx-e*0.3,y-e,cw+e*0.6,h+2*e,d+e   # front + top only (opaque), so neighbours stay unchanged
    ax.add_patch(Polygon([[X,Y+H],[X+W,Y+H],[X+W+D*DX,Y+H+D*DY],[X+D*DX,Y+H+D*DY]],fc="#8fc0ee",ec=INK,lw=1.0,zorder=7))
    ax.add_patch(Polygon([[X,Y],[X+W,Y],[X+W,Y+H],[X,Y+H]],fc=CH[HL],ec=INK,lw=1.0,zorder=8))
    return gx,cw,y+h,d
def box(ax,x,y,w,h,t):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",fc="#f4f4f1",ec="#888",lw=0.9))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=9,color=INK)
def arrow(ax,a,b,lab=None,off=(0,0.08)):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=8,lw=0.9,color="#555"))
    if lab: ax.text((a[0]+b[0])/2+off[0],(a[1]+b[1])/2+off[1],lab,ha="center",va="bottom",fontsize=8,color=MUT)
def wmat(ax,x,y,cw=0.11,h=0.45):
    for i,c in enumerate(CH): ax.add_patch(Rectangle((x+i*cw,y),cw,h,fc=c,ec="white",lw=0.6))
    e=0.05; ax.add_patch(Rectangle((x+HL*cw-e*0.4,y-e),cw+e*0.8,h+2*e,fc=CH[HL],ec=INK,lw=1.0,zorder=5))
fig,ax=plt.subplots(figsize=(5.6,2.5))
# unit U with producers
ax.add_patch(FancyBboxPatch((-0.1,0.12),0.95,2.0,boxstyle="round,pad=0.02,rounding_size=0.08",fill=False,ec=MUT,lw=0.9,ls=(0,(3,2))))
ax.text(0.37,2.2,"$U$",ha="center",va="bottom",fontsize=9,color=MUT)
box(ax,0.02,1.45,0.72,0.48,"$\\div s$"); box(ax,0.02,0.3,0.72,0.48,"$\\div s$")
arrow(ax,(0.76,1.69),(1.2,1.25)); arrow(ax,(0.76,0.54),(1.2,0.98))
ax.add_patch(Circle((1.33,1.12),0.12,fc="white",ec="#555",lw=0.9)); ax.text(1.33,1.12,"+",ha="center",va="center",fontsize=9,color=INK)
arrow(ax,(1.46,1.12),(1.7,1.12))
gx,cw,ytop,dd=tensor(ax,1.72,0.82)
ax.text(1.72+0.25,0.72,"$x/s$",ha="center",va="top",fontsize=9,color=INK)
ax.text(gx+0.02,ytop+dd*DY+0.22,"$s_g$",ha="center",va="bottom",fontsize=9,color=ACC)
# consumers
arrow(ax,(2.5,1.25),(2.85,1.68),lab="$x^{(1)}$",off=(-0.12,0.05)); arrow(ax,(2.5,1.0),(2.85,0.55),lab="$x^{(2)}$",off=(-0.13,-0.27))
wmat(ax,2.9,1.5); ax.text(3.5,1.72,"$W_1\\,\\mathrm{diag}(s)$",va="center",fontsize=9,color=INK)
wmat(ax,2.9,0.33); ax.text(3.5,0.55,"$W_2\\,\\mathrm{diag}(s)$",va="center",fontsize=9,color=INK)
ax.plot([4.55,4.62,4.62,4.55],[1.95,1.95,0.33,0.33],color=ACC,lw=0.9)
ax.text(4.68,1.14,"$\\mathcal{C}(g)$",ha="left",va="center",fontsize=9,color=ACC)
ax.set_xlim(-0.2,5.2); ax.set_ylim(0.0,2.35); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
for suf,kw in (("",dict(dpi=220)),("_transparent",dict(dpi=300,transparent=True))):
    fig.savefig(OUT+f"producer_consumer_min{suf}.png",bbox_inches="tight",**kw)
fig.savefig(OUT+"producer_consumer_min.pdf",bbox_inches="tight"); print("ok")
