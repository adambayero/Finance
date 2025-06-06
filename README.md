# 💰 Financial Instruments Pricing & Volatility Modeling

Projet Python orienté finance quantitative pour le pricing d'instruments de taux (obligations, swaps, swaptions), la construction de courbes de taux, la calibration de modèles (SABR), et la visualisation de surfaces/cubes de volatilité implicite.

## ✨ Fonctionnalités

- **Pricing d'obligations** (fixed, floating, zero-coupon, inflation)
- **Pricing de swaps**
- **Pricing de swaptions via Black-76**
- **Calibration du modèle SABR** à partir de vols implicites de marché
- **Simulation de la dynamique du modèle SABR**
- **Affichage de surfaces et cubes de volatilité**
- **Bootstrap de courbe zéro-coupon**

## 🛠️ Installation

```bash
git clone https://github.com/<your-username>/finance-project.git
cd finance-project
pip install -r requirements.txt
```

## 🚀 Utilisation
Lancer le script principal :

```bash
./run.bat
```

Ou directement :

```bash
python scripts/main.py
```

## 📊 Exemples

- display_3d_graph : surface de volatilité (strike vs maturity vs vol)
- display_cube : cube de volatilité (strike, maturity, tenor)
- display_dynamic_grid : animation de la dynamique stochastique de F_t, alpha_t
​
## 📎 Auteurs

Projet développé dans un contexte de R&D en finance quantitative.
Réalisé par Tom Dupont supervisé par Mahdi Akkouh