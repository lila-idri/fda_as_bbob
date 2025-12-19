import cocoex # Import du Benchmark COCO

from src.benchmark_algorithms import  fda_baseline, fda_wsc, fractop, soo, direct


# Liste de tous les algorithmes à tester/comparer
algorithms_to_test = {
    "FDA": fda_baseline.fda_optimizer,
    "FDA-AS": fda_wsc.fda_wsc_optimizer,
    #"DIRECT": direct.direct_optimizer,
    #"FRACTOP": fractop.fractop_optimizer,
    #"SOO": soo.soo_optimizer,
}


def run_optimizer_on_suite(optimizer_func_, suite_name="bbob", output_folder=""):
    # Création de l'observateur BBOB :  configure COCO pour enregistrer les résultats
    observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)

    # Chargement de la suite de fonctions
    suite = cocoex.Suite(suite_name, "year:2025", "dimensions: 2,3,5,10,20,40")  # Exemple de dimensions

    for problem in suite:
        # Enregistrer les résultats de l'exécution
        problem.observe_with(observer)

        # NB_EVAL_MAX
        budget = 20000 * problem.dimension

        # Exécution de l'algorithme
        optimizer_func_(problem, problem.dimension, budget)


if __name__ == "__main__":
    for name, optimizer_func in algorithms_to_test.items():
        print(f"Lancement du benchmarking pour : {name}")

        # Sous-dossier pour chaque algo
        output_dir = f"{name}"

        # Exécution sur la suite BBOB
        run_optimizer_on_suite(
            optimizer_func,
            suite_name="bbob",
            output_folder=output_dir
        )

    print("Benchmarking COCO/BBOB terminé pour tous les algorithmes.")