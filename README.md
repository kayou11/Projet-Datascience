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
<a href="https://github.com/medouledou"><img src="https://avatars2.githubusercontent.com/u/19546375?s=400&v=4" title="Medouledou" width="80" height="80"></a>
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
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/clean.jpg" width="200" title="image clean">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/degraded.jpg" width="200" title="image degraded">
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
  <img src="https://miro.medium.com/max/1428/1*M2Er7hbryb2y0RP1UOz5Rw.png" width="450" title="schéma d'un GAN">
</p>
<br><br>

### Entrainement
L'entrainement de ce model est assez long et délicat à réaliser puisqu'il sagit d'entrainer deux éléments que sont **le discriminateur** et le **générateur**.
Le générateur doit être capable de créer des images pour pouvoir tromper le discriminateur et le discriminateur doit être capable de bien distinguer les fausses images.

Notre entrainement doit être donc assez long pour generer des images correct, mais pas trop non plus puisque nous ne disposons pas de puissance de calculs nécessaire pour faire des entrainements de plus de 100 epochs.

Nous avons donc choisi pour la démonstration, d'entrainer notre modèle sur 50 epochs et un batch de 32, ce qui correspond à environ 2h30 de calculs.<br />
A la première epoch nous pouvons observer visuellement que le generateur a créé une image plus dégradée encore que l'image dégradée que l'on avait de base.
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/original.png" width="200" title="Image originale"> 
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/degrade.png" width="200" title="Image dégradée">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/generated.png" width="200" title="Image générée">
</p>
<br><br>

Cela est confirmé par la mesure ci-dessous, le SSIM (Structural Similarity Index) qui nous sert à calculer la différence de structure entre deux images.<br />
La mesure sort des résultats entre -1 et 1. La valeur 1 correspond à deux images identiques et la valeur 0 à aucune similitude.
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/ssim1.png" width="400" title="image originale">
</p>
<br><br>

Au bout de l'entrainement, à la 48e epoch, nous pouvons voir que notre générateur à créer une image qui visuellement se rapproche de l'image réelle même si il reste encore un peu de dégradation.
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/original1.png" width="200" title="Image originale"> 
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/degrade2.png" width="200" title="Image dégradée">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/generated1.png" width="200" title="Image générée">
</p>
<br><br>

Et la mesure SSIM nous montre que l'image que nous avons générée est moins dégradée mathématiquement parlant.
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/ssim2.png" width="400" title="image originale">
</p>
<br><br>

Après cet entraînement, nous pouvons dire que notre modèle n'est pas encore parfait et qu'il reste du travail.<br />
Mais avec une optimisation au niveau des hyperparamètres (nombre d'epoch, taille du batch), du code et/ou avec une plus grosse puissance de calculs, nous pourrions avoir un modèle plus performant.

## Workflow Entrainement

Pour utiliser le workflow d'entrainement, il faut télécharger le fichier **Workflow_Entrainement_Model.ipynb** ci-dessus et l'ouvrir avec <a href="https://colab.research.google.com/notebooks/intro.ipynb#recent=true">Google Colab</a>.<br />
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/colab.PNG" title="Google Colab">
</p>
<br><br>
Ensuite, vous arriverez sur la page du notebook que vous pourrez executer en cliquant sur l'onglet "Execution" puis sur "Tout Executer".<br />
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/colab2.PNG" title="Google Colab">
</p>
<br><br>


## Workflow Utilisation

De la même façon que le workflow d'entrainement, pour utiliser le workflow d'utilisation, il faut téléchargé le fichier **Workflow_d'utilisation_Photo.ipynb** ci-dessus et l'ouvrir avec <a href="https://colab.research.google.com/notebooks/intro.ipynb#recent=true">Google Colab</a>.<br />
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/colab.PNG" title="Google Colab">
</p>
<br><br>

Ensuite, avant d'executer le notebook, il faut importer les images à améliorer.<br />
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/import.png" title="Google Colab">
</p>
<br><br>

Puis, il suffit maintenant d'indiquer dans la dernière cellule du notebook, l'emplacement des images et d'executer tout le notebook (Cf. la partie exécution du Workflow d'entrainement).<br />
<p align="center">
  <img src="https://github.com/kayou11/Projet-Datascience/blob/master/img-readme/utilisation.png" title="Google Colab">
</p>
<br><br>
