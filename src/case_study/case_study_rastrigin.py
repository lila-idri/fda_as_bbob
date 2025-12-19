import numpy as np
import matplotlib.pyplot as plt

# =========================
# Fonction objectif
# =========================
def f_(x, y):
    return (x-1)**2 + (y-1)**2 + 0.5*np.sin(5*x)*np.sin(5*y)

# =========================
# Shifted Rastrigin
#==========================
x0, y0 = 1.5, -0.5
def f(x, y):
    x0, y0 = 3, -0.5   # décalage du minimum global
    a = 1
    return (
        a
        + (x - x0)**2 + (y - y0)**2
        - a*(np.cos(2*np.pi*(x - x0)) + np.cos(2*np.pi*(y - y0)))
    )

# =========================
# Domaine
# =========================
xmin, xmax = -2, 2
ymin, ymax = -2, 2


x = np.linspace(xmin, xmax, 400)
y = np.linspace(ymin, ymax, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# =========================
# Sphère parent
# =========================
center = np.array([0.0, 0.0])
R = (xmax - xmin) / 2

# =========================
# FDA geometry (exacte)
# =========================
delta = 1 + np.sqrt(2)
r_child = R / delta
offset = R - r_child

child_centers = [
    center + np.array([ offset, 0.0]),
    center + np.array([-offset, 0.0]),
    center + np.array([0.0,  offset]),
    center + np.array([0.0, -offset])
]

# =========================
# Dimension
# =========================
D = 2

# =========================
# BSF volontairement biaisé
# =========================
#
BSF = np.array([-1.5, -1.1])

# =========================
# Global Min
# =========================
xg, yg = x0, y0

# =========================
# Calcul PHS et AS
# =========================
results = []
fda_points = []   # pour le plot

def dist(a, b):
    return np.linalg.norm(a - b)

# paramètres AS (exemple)
l = 1
k = 5
w2 = l / k
w1 = 1 - w2

for j, Cj in enumerate(child_centers):

    # FDA points (formule exacte que TU as donnée)
    s1 = Cj + (r_child / np.sqrt(D)) * np.ones(D)
    s2 = Cj - (r_child / np.sqrt(D)) * np.ones(D)

    fda_points.append((Cj, s1, s2))

    # Valeurs de la fonction
    fC  = f(Cj[0],  Cj[1])
    fs1 = f(s1[0], s1[1])
    fs2 = f(s2[0], s2[1])

    # ---------- PHS (formule FDA correcte) ----------
    g_c  = fC  / dist(Cj, BSF)
    g_s1 = fs1 / dist(s1, BSF)
    g_s2 = fs2 / dist(s2, BSF)

    PHS = max(g_c, g_s1, g_s2)

    # ---------- AS (comme TU l'as défini) ----------
    vals = np.array([fC, fs1, fs2])
    sigma = np.std(vals)

    q_c = w1 * sigma - w2 * fC
    q_s1 = w1 * sigma - w2 * fs1
    q_s2 = w1 * sigma - w2 * fs2

    AS = max(q_c, q_s1, q_s2)

    results.append({
        "sphere": j + 1,
        "center": Cj,
        "PHS": PHS,
        "AS": AS
    })



# =========================
# Affichage tableau
# =========================
print("\nSphere | Center               |   PHS     |    AS")
print("----------------------------------------------------")
for r in results:
    print(f"{r['sphere']:>6} | {r['center']} | {r['PHS']:.4f} | {r['AS']:.4f}")

# =========================
# Visualisation
# =========================
plt.figure(figsize=(5, 5))

# Contours
plt.contour(X, Y, Z, levels=20, colors='grey', zorder=0)

# Sphère parent
plt.gca().add_patch(
    plt.Circle(center, R, fill=False, linewidth=2, color='black')
)

# Sphères enfants + etiquettes
for j, Cj in enumerate(child_centers):
    if j==3:
        plt.gca().add_patch(
            plt.Circle(Cj, r_child, fill=False, linewidth=3, linestyle='-', color='green')
        )
    elif j==1:
        plt.gca().add_patch(
            plt.Circle(Cj, r_child, fill=False, linewidth=3, linestyle='-', color='red')
        )
    else:
        plt.gca().add_patch(
            plt.Circle(Cj, r_child, fill=False, linewidth=2, linestyle='--', color='blue')
        )
    plt.text(
        Cj[0] + 0.05, Cj[1] + 0.05,
        f"S{j+1}", fontsize=15, weight='bold'
    )

# Plot des points échantillonnés
for i, (Cj, s1, s2) in enumerate(fda_points):
    P = np.vstack([Cj, s1, s2])
    plt.scatter(
        P[:, 0], P[:, 1],
        c='#FF8C00', edgecolors='black',
        s=25,
        zorder=7,
        label='Sampled points' if i == 0 else None
    )

# Points FDA (centre, s1, s2)
for Cj, s1, s2 in fda_points:
    P = np.vstack([Cj, s1, s2])
    plt.scatter(P[:, 0], P[:, 1], c='blue', s=25, zorder=6)

best_phs_idx = np.argmax([r["PHS"] for r in results])
best_as_idx  = np.argmax([r["AS"]  for r in results])

C_phs = results[best_phs_idx]["center"]
C_as  = results[best_as_idx]["center"]




# Lien : PHS ->minimum global
plt.plot(
    [C_phs[0], xg], [C_phs[1], yg],
    linestyle='-', linewidth=2, color='red',
    label='PHS selection'
)

# Lien : AS --> minimum global
plt.plot(
    [C_as[0], xg], [C_as[1], yg],
    linestyle='-', linewidth=2, color='green',
    label='AS selection'
)

# BSF
plt.scatter(BSF[0], BSF[1], c='purple', s=80, label='BSF', zorder=6)

# Minimum global (référence)
idx = np.argmin(Z)
xg, yg = X.flat[idx], Y.flat[idx]
plt.scatter(xg, yg, c='green', s=80, label='Global Min', zorder=6)

plt.gca().set_aspect('equal')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.legend(loc='upper left', ncol=3,  bbox_to_anchor=(-0.15, -0.13), frameon=False)
plt.title("2-D Shifted Rastrigin")
plt.grid(True, alpha=0.3)

plt.savefig("f1.pdf", dpi=300, bbox_inches="tight")
plt.close()
