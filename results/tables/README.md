# Tables

Produced by the notebooks in `../../notebooks/`. File names and column headers are in Spanish,
since that is what the code writes; the descriptions below are the English equivalents.

`tablas_para_el_articulo.md` collects the four main tables already formatted for the manuscript.
Everything else is the underlying data.

## From `00_deduplication.ipynb`

| File | Contents |
| --- | --- |
| `tabla_dedup_metodos.csv` | Duplicate detection methods compared: pairs found by each, how many the embedding detector supports, and how many of the unsupported ones join images from different classes |
| `tabla_barrido_umbral.csv` | Cosine threshold sweep from 0.95 to 0.999, with the number of validation and test images flagged at each |
| `tabla_fuga_por_clase.csv` | Per class: training images kept and discarded, test images with a duplicate in training |
| `pares_etiqueta_contradictoria.csv` | The 187 near-identical pairs whose labels disagree, with file paths |

## From `02_evaluation.ipynb`

| File | Contents |
| --- | --- |
| `tabla_comparativa_arquitecturas.csv` | The six architectures: accuracy, macro-F1, weighted-F1, AUC, parameters, GFLOPs, latency, training time |
| `tabla_mcnemar.csv` | Paired significance tests of each architecture against ConvNeXt-S + CBAM |
| `tabla_sin_fuga.csv` | Metrics on the full test set and on the subset with no detected duplicate in training |
| `tabla_cambio_por_clase_sin_fuga.csv` | Per-class F1 before and after removing the contaminated test images |
| `tabla_incertidumbre_por_clase.csv` | Per-class precision and recall with Wilson intervals, F1 with bootstrap intervals |
| `tabla_ablacion_por_clase.csv` | Per-class F1 with and without CBAM |
| `principales_confusiones.csv` | The fifteen most frequent confusions between classes |
| `matriz_confusion.csv` | Raw confusion matrix, counts |
| `curva_umbral.csv` | Coverage and precision over accepted predictions at every confidence threshold, on calibrated probabilities |
| `resumen_fase2.json` | Summary numbers, including the ablation tests and calibration error |
