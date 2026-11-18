import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch, Circle
OUT="/tmp/claude-1003/-home-taeho/5692497e-a15d-476f-95db-3ec3f5583e11/scratchpad/fig_pc/"
CH=["#2a78d6","#5fa8e8","#9cc9f0","#f2a65a","#d9603b"]   # per-channel scale s_c (one color per channel)
INK="#222"; MUT="#6b6a64"
def cuboid(ax,x,y,w=0.55,h=0.9,d=0.25,label=None,alpha=1):
    n=len(CH); cw=w/n
    for i,c in enumerate(CH):   # front faces = channels
        x0=x+i*cw
        ax.add_patch(Polygon([[x0,y],[x0+cw,y],[x0+cw,y+h],[x0,y+h]],fc=c,ec="white",lw=0.6,alpha=alpha))
        ax.add_patch(Polygon([[x0,y+h],[x0+cw,y+h],[x0+cw+d,y+h+d*0.6],[x0+d,y+h+d*0.6]],fc=c,ec="white",lw=0.6,alpha=alpha*0.75))
    ax.add_patch(Polygon([[x+w,y],[x+w+d,y+d*0.6],[x+w+d,y+h+d*0.6],[x+w,y+h]],fc=CH[-1],ec="white",lw=0.6,alpha=alpha*0.55))
    if label: ax.text(x+w/2+d/2,y-0.13,label,ha="center",va="top",fontsize=7.5,color=INK)
def box(ax,x,y,w,h,text,fc="#f4f4f1",ec="#888"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",fc=fc,ec=ec,lw=0.9))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=7.5,color=INK)
def wcols(ax,x,y,label):  # consumer weight with per-input-channel columns colored by the same s
    n=len(CH); cw=0.12
    for i,c in enumerate(CH):
        ax.add_patch(Polygon([[x+i*cw,y],[x+(i+1)*cw,y],[x+(i+1)*cw,y+0.55],[x+i*cw,y+0.55]],fc=c,ec="white",lw=0.6))
    ax.text(x+n*cw+0.08,y+0.27,label,ha="left",va="center",fontsize=7.5,color=INK)
def arrow(ax,a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=8,lw=0.9,color="#555"))

fig,axs=plt.subplots(1,2,figsize=(7.2,2.3),gridspec_kw=dict(width_ratios=[1,1.15]))
# (a) fan-out
ax=axs[0]
box(ax,0.0,0.75,0.95,0.55,"producer conv\n+ requant $\\div s$")
arrow(ax,(0.97,1.02),(1.25,1.02))
cuboid(ax,1.28,0.58,label="$x/s$  (one tensor,\nper-channel $s_c$)")
arrow(ax,(2.1,1.25),(2.55,1.75)); arrow(ax,(2.1,0.85),(2.55,0.35))
wcols(ax,2.6,1.55,"$W_1\\,\\mathrm{diag}(s)$"); wcols(ax,2.6,0.1,"$W_2\\,\\mathrm{diag}(s)$")
ax.text(3.3,-0.35,"consumers $\\mathcal{C}(g)$ share $s$",ha="center",fontsize=7,color=MUT)
ax.set_title("(a) fan-out: one producer, many consumers",fontsize=8,loc="left",color=INK)
# (b) residual add
ax=axs[1]
box(ax,0.0,1.45,0.95,0.5,"producer A\n$\\div s$"); box(ax,0.0,0.15,0.95,0.5,"producer B\n$\\div s$")
arrow(ax,(0.97,1.7),(1.45,1.2)); arrow(ax,(0.97,0.4),(1.45,0.9))
ax.add_patch(Circle((1.6,1.05),0.15,fc="white",ec="#555",lw=0.9)); ax.text(1.6,1.05,"+",ha="center",va="center",fontsize=10,color=INK)
arrow(ax,(1.76,1.05),(2.05,1.05))
cuboid(ax,2.08,0.6,label="same $s$ on both\nbranches (tied group $g$)")
arrow(ax,(2.9,1.05),(3.3,1.05)); wcols(ax,3.35,0.8,"$W\\,\\mathrm{diag}(s)$")
ax.set_title("(b) residual add: producers tied to one $s$",fontsize=8,loc="left",color=INK)
for a in axs: a.set_xlim(-0.1,4.3); a.set_ylim(-0.5,2.25); a.set_aspect("equal"); a.axis("off")
fig.tight_layout(); fig.savefig(OUT+"producer_consumer_draft.png",dpi=220,bbox_inches="tight"); fig.savefig(OUT+"producer_consumer_draft.pdf",bbox_inches="tight")
print("ok")
