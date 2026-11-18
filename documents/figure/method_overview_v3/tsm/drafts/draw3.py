import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
OUT="/tmp/claude-1003/-home-taeho/5692497e-a15d-476f-95db-3ec3f5583e11/scratchpad/fig_pc/"
CH=["#2a78d6","#5fa8e8","#9cc9f0","#f2a65a","#d9603b"]; INK="#222"; MUT="#6b6a64"; ACC="#b03a2e"
HL=1  # highlighted channel = one group g
def strip(ax,x,y,w,h,hl=True):
    cw=w/len(CH)
    for i,c in enumerate(CH):
        ax.add_patch(Polygon([[x+i*cw,y],[x+(i+1)*cw,y],[x+(i+1)*cw,y+h],[x+i*cw,y+h]],fc=c,ec="white",lw=0.6))
    if hl: ax.add_patch(Rectangle((x+HL*cw,y),cw,h,fill=False,ec=INK,lw=1.1,zorder=5))
    return cw
def cuboid(ax,x,y,w=0.6,h=0.95,d=0.25):
    cw=strip(ax,x,y,w,h)
    for i,c in enumerate(CH):
        x0=x+i*cw; ax.add_patch(Polygon([[x0,y+h],[x0+cw,y+h],[x0+cw+d,y+h+d*0.6],[x0+d,y+h+d*0.6]],fc=c,ec="white",lw=0.6,alpha=0.75))
    ax.add_patch(Polygon([[x+w,y],[x+w+d,y+d*0.6],[x+w+d,y+h+d*0.6],[x+w,y+h]],fc=CH[-1],ec="white",lw=0.6,alpha=0.55))
    return cw
def box(ax,x,y,w,h,t):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",fc="#f4f4f1",ec="#888",lw=0.9))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=7.3,color=INK)
def arrow(ax,a,b,**k): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=8,lw=0.9,color=k.get("c","#555"),connectionstyle=k.get("cs","arc3")))
fig,ax=plt.subplots(figsize=(6.4,2.9))
# unit U (dashed) around producers
ax.add_patch(FancyBboxPatch((-0.12,0.1),1.2,2.15,boxstyle="round,pad=0.02,rounding_size=0.08",fill=False,ec=MUT,lw=0.9,ls=(0,(3,2))))
ax.text(0.48,2.36,"unit $U$: one $\\alpha^*_U$ (Eq. 12)",ha="center",fontsize=7,color=MUT)
box(ax,0.0,1.55,0.95,0.55,"producer A\nrequant $\\div s$"); box(ax,0.0,0.25,0.95,0.55,"producer B\nrequant $\\div s$")
arrow(ax,(0.97,1.82),(1.52,1.32)); arrow(ax,(0.97,0.52),(1.52,1.08))
ax.add_patch(Circle((1.67,1.2),0.15,fc="white",ec="#555",lw=0.9)); ax.text(1.67,1.2,"+",ha="center",va="center",fontsize=10,color=INK)
ax.text(1.78,0.98,"residual\nadd",ha="center",va="top",fontsize=6.5,color=MUT)
arrow(ax,(1.83,1.2),(2.13,1.2))
cw=cuboid(ax,2.15,0.72)
ax.text(2.45,0.6,"$x/s$  (one tensor)",ha="center",va="top",fontsize=7.2,color=INK)
# group g annotation on highlighted channel
gx=2.15+HL*cw+cw/2
ax.annotate("group $g$ (one channel, or\nchannels merged by the add):\none scale $s_g$; channel $c$ uses $s_{g(c)}$",
            xy=(gx,1.67+0.1),xytext=(1.95,2.42),fontsize=6.5,color=ACC,ha="left",va="bottom",
            arrowprops=dict(arrowstyle="-",color=ACC,lw=0.8))
# consumers
arrow(ax,(3.02,1.45),(3.42,1.9)); arrow(ax,(3.02,0.98),(3.42,0.52))
strip(ax,3.47,1.7,0.6,0.5); ax.text(4.15,1.95,"consumer $j{=}1$:  $W_1\\,\\mathrm{diag}(s)$",va="center",fontsize=7.2,color=INK)
strip(ax,3.47,0.25,0.6,0.5); ax.text(4.15,0.5,"consumer $j{=}2$:  $W_2\\,\\mathrm{diag}(s)$",va="center",fontsize=7.2,color=INK)
ax.text(4.15,1.6,"input $x^{(1)}$",va="top",fontsize=6.3,color=MUT); ax.text(4.15,0.15,"input $x^{(2)}$",va="top",fontsize=6.3,color=MUT)
ax.text(5.05,1.22,"$j\\in\\mathcal{C}(g)$ share $s_g$",ha="center",va="center",fontsize=7,color=ACC)
# bottom: statistics (Eq. 10-11) and migration (Eq. 9)
ax.text(2.6,-0.12,"stats per group: $\\mathrm{max}_{j\\in\\mathcal{C}(g)}\\,\\widetilde{x}^{(j)}_c$ with floor $\\epsilon$ (Eqs. 10, 11);  $s_c(\\alpha)$ by Eq. 9",
        ha="center",va="top",fontsize=6.6,color=MUT)
ax.text(2.6,-0.42,"same color = same scale; $\\div s$ is folded into the producer's requant (no run-time op)",ha="center",va="top",fontsize=6.6,color=MUT)
ax.set_xlim(-0.2,6.0); ax.set_ylim(-0.75,2.95); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
for suf,kw in (("",dict(dpi=220)),("_transparent",dict(dpi=300,transparent=True))):
    fig.savefig(OUT+f"producer_consumer_annot{suf}.png",bbox_inches="tight",**kw)
fig.savefig(OUT+"producer_consumer_annot.pdf",bbox_inches="tight"); print("ok")
