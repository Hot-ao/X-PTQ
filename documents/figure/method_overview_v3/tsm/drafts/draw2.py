import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch, Circle
OUT="/tmp/claude-1003/-home-taeho/5692497e-a15d-476f-95db-3ec3f5583e11/scratchpad/fig_pc/"
CH=["#2a78d6","#5fa8e8","#9cc9f0","#f2a65a","#d9603b"]; INK="#222"; MUT="#6b6a64"
def strip(ax,x,y,w,h):
    cw=w/len(CH)
    for i,c in enumerate(CH): ax.add_patch(Polygon([[x+i*cw,y],[x+(i+1)*cw,y],[x+(i+1)*cw,y+h],[x+i*cw,y+h]],fc=c,ec="white",lw=0.6))
def cuboid(ax,x,y,w=0.55,h=0.9,d=0.25):
    strip(ax,x,y,w,h); cw=w/len(CH)
    for i,c in enumerate(CH):
        x0=x+i*cw; ax.add_patch(Polygon([[x0,y+h],[x0+cw,y+h],[x0+cw+d,y+h+d*0.6],[x0+d,y+h+d*0.6]],fc=c,ec="white",lw=0.6,alpha=0.75))
    ax.add_patch(Polygon([[x+w,y],[x+w+d,y+d*0.6],[x+w+d,y+h+d*0.6],[x+w,y+h]],fc=CH[-1],ec="white",lw=0.6,alpha=0.55))
def box(ax,x,y,w,h,t):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",fc="#f4f4f1",ec="#888",lw=0.9))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=7.5,color=INK)
def arrow(ax,a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=8,lw=0.9,color="#555"))
fig,ax=plt.subplots(figsize=(5.4,2.3))
# producers (tied by residual add)
box(ax,0.0,1.55,0.95,0.5,"producer A\nrequant $\\div s$"); box(ax,0.0,0.25,0.95,0.5,"producer B\nrequant $\\div s$")
arrow(ax,(0.97,1.8),(1.47,1.27)); arrow(ax,(0.97,0.5),(1.47,1.03))
ax.add_patch(Circle((1.62,1.15),0.15,fc="white",ec="#555",lw=0.9)); ax.text(1.62,1.15,"+",ha="center",va="center",fontsize=10,color=INK)
arrow(ax,(1.78,1.15),(2.08,1.15))
cuboid(ax,2.1,0.7)
ax.text(2.48,0.55,"$x/s$: one tensor,\nper-channel $s_c$",ha="center",va="top",fontsize=7.2,color=INK)
# consumers (fan-out)
arrow(ax,(2.95,1.4),(3.35,1.85)); arrow(ax,(2.95,0.95),(3.35,0.5))
strip(ax,3.4,1.65,0.6,0.5); ax.text(4.08,1.9,"$W_1\\,\\mathrm{diag}(s)$",va="center",fontsize=7.5,color=INK)
strip(ax,3.4,0.2,0.6,0.5); ax.text(4.08,0.45,"$W_2\\,\\mathrm{diag}(s)$",va="center",fontsize=7.5,color=INK)
# brackets / captions
ax.text(0.48,0.02,"producers tied to one $s$\n(residual add)",ha="center",va="top",fontsize=7,color=MUT)
ax.text(4.25,0.02,"consumers $\\mathcal{C}(g)$ share $s$\n(fan-out)",ha="center",va="top",fontsize=7,color=MUT)
ax.text(2.48,2.35,"same color = same scale; folded into requant, no run-time op",ha="center",fontsize=7,color=MUT)
ax.set_xlim(-0.1,5.1); ax.set_ylim(-0.45,2.5); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT+"producer_consumer_merged.png",dpi=220,bbox_inches="tight",transparent=False)
fig.savefig(OUT+"producer_consumer_merged.pdf",bbox_inches="tight"); fig.savefig(OUT+"producer_consumer_merged_transparent.png",dpi=300,bbox_inches="tight",transparent=True)
print("ok")
