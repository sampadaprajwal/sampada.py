import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
import igraph
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
import igraph

adata = sc.datasets.pbmc3k()

print(adata)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt"],
    inplace=True
)
adata.obs.head()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(adata, max_value=10)
sc.pl.pca(adata)
adata.write("processed_data.h5ad")
adata.write(r"C:\Users\Prajwal\Desktop\processed_data.h5ad")