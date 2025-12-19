import cocopp
import os

input_parent_dir = 'exdata'

# Liste des dossiers à analyser
input_dirs = [
    os.path.join(input_parent_dir, 'FDA'),
    os.path.join(input_parent_dir, 'FDA-AS'),
    os.path.join(input_parent_dir, 'DIRECT'),
    os.path.join(input_parent_dir, 'FRACTOP'),
    os.path.join(input_parent_dir, 'SOO'),
]
output_dir = '../../results/bbob_post_processed'
# Crée les dossiers si nécessaire
os.makedirs(output_dir, exist_ok=True)

print("Lancement de l'analyse cocopp...")
# L'exécution du post-traitement
cocopp.main(input_dirs)

print(f"Analyse terminée. Les résultats (graphes et HTML) sont dans : {output_dir}")