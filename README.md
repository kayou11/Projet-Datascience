<p align="center">
  <img src="http://www.alsacetech.org/wp-content/uploads/2018/12/Cesi_Logo_INGENIEUR_quadri.png" width="350" title="hover text">
</p>

<h3 align="center">Projet Datascience</h3>
<p align="center">
  :bar_chart: CESI Bordeaux - Options <strong>Data Science</strong>!
  <br><br>
</p>

## Contributeurs
[//]: contributor-faces

<a href="https://github.com/Pielgrin"><img src="https://avatars3.githubusercontent.com/u/18363758?s=400&v=4" title="Pielgrin" width="80" height="80"></a>
<a href="https://github.com/Popclem"><img src="https://avatars1.githubusercontent.com/u/19546378?s=400&v=4" title="Popclem" width="80" height="80"></a>
<a href="https://github.com/clement-BRE"><img src="https://avatars3.githubusercontent.com/u/19546525?s=400&v=4" title="Clement-Bre" width="80" height="80"></a>
<a href="https://github.com/kayou11"><img src="https://avatars1.githubusercontent.com/u/16228196?s=460&v=4" title="Kayou" width="80" height="80"></a>

[//]: contributor-faces

## Contents

- [Context](#context)
- [Le Projet](#le-projet)
- [Workflow Entrainement](#workflow-entrainement)
- [Workflow Utilisation](#workflow-utilisation)

## Context

L’entreprise TouNum travaille sur la numérisation des vieilles cassettes vidéo. Des historiens ont fait
appel à eux pour un projet de recherche de la plus haute importance sur l’analyse de certaines vidéos
historiques au format PAL qu’ils détiennent sur des cassettes vidéo VHS. Ils ont besoin d’une version
numériques de haute qualité d’image pour leur projet. Or, ces images nécessitent une forte
amélioration. TouNum aimerait, dans un premier temps, répondre à cette demande qui nécessite de
la restauration d’images mais pas nécessairement de vidéos dans leur ensemble, pour ensuite
adapter son produit pour toucher un plus grand public en restaurant des vidéos.

On nous propose un premier contrat pour travailler sur une solution visant à améliorer la qualité
d’images PAL issues de la numérisation de vidéos stockées sur cassettes VHS. Ces images ont non
seulement une faible résolution (avec des pixels non carrés, ce qui n’aide pas), mais elles ont de
nombreux problèmes de qualité (grain analogique, top-screen tearing, jitter horizontal, aberrations
chromatiques, sous-échantillonnage de la chrominance et de la luminance…), qui apparaissent de
manière plus ou moins aléatoire.

## Le Projet
### Data

Les images qui nous ont été fournies proviennent du dataset COCO.
Un dataset est **clean**, l'autre **degraded** comme cité ci-dessus et comporte chacun 4500 images.

Nous nous sommes servis du dataset **clean** pour entrainer notre model. 
En effet, nous avons décidé de dégrader nous même les images à la volée selon le nombre d'entrainement **(epochs)** que nous souhaitions effectuer mais aussi pour faire correspondre correctement les images entre elles.

Nous avons donc construit une classe permettant de dégrader une image de manière à se rapprocher des dégradations présentent sur les images du dataset **degraded**.
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/clean.jpg" width="200" title="hover text">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/degraded.jpg" width="200" title="hover text">
</p>
<br><br>

### Model
Pour le model, nous avons choisi d'implémenter **Pix2pix** un model GAN (Generative Adversarial Networks).
L'idée derrière un GAN est que l'on a deux réseaux, un **Générateur** et un **Discriminateur**, en concurrence l'un avec l'autre.<br />
Le générateur fabrique de fausses données à transmettre au discriminateur. Le discriminateur voit également les données réelles et prédit si les données qu'il reçoit sont réelles ou fausses.<br />
Le générateur est entraîné pour tromper le discriminateur, il veut produire des données qui ressemblent le plus possible à des données réelles. Et le discriminateur est entraîné pour savoir quelles données sont réelles et quelles données sont fausses.<br /> 
En fin de compte, le générateur apprend à fournir au discriminateur des données idéalement impossibles à distinguer des données réelles.
<br><br>
<p align="center">
  <img src="https://miro.medium.com/max/1428/1*M2Er7hbryb2y0RP1UOz5Rw.png" width="450" title="hover text">
</p>
<br><br>

### Entrainement
Pour cette exemple, nous avons entrainé le model sur 100 epochs avec un batch_size de 32.<br />
A la 63e epoch nous constatons que nous avons une amélioration visuelle de notre image.
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/before.PNG" width="200" title="hover text">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/after.PNG" width="200" title="hover text">
</p>
<br><br>

Cependant lorsque nous affichons les distances de Manhattan (L1) et Euclidienne (L2), nous constatons que les images se sont en réalité détériorées.
<br><br>
![Metrics](https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/metrics.PNG)
<br><br>
Et nous constatons également qu'à la fin de l'entrainement, sur ces deux même distances, les images se sont en moyenne plus dégradées qu'améliorées.
<br><br>
![Metrics](https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/metrics2.PNG)

## Le Workflow Entrainement

Pour utiliser le workflow d'entrainement, il faut télécharger le fichier **Workflow_Entrainement_Model.ipynb** ci-dessus et l'ouvrir avec <a href="https://colab.research.google.com/notebooks/intro.ipynb#recent=true">Google Colab</a>.<br />
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/colab.PNG" title="hover text">
</p>
<br><br>
Ensuite, vous arriverez sur la page du notebook que vous pourrez executer en cliquant sur l'onglet "Execution" puis sur "Tout Executer".<br />
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/colab2.PNG" title="hover text">
</p>
<br><br>


## Le Workflow Utilisation
