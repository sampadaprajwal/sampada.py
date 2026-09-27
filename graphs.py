import pandas as pd
data = pd.read_excel("C:/Users/Prajwal/Downloads/gene_expression_cancer_classification.xlsx")
import matplotlib.pyplot as plt
print(data.head())
X = data.drop(columns=["Sample_ID", "Target"])
print(data.head())
print(X.head())
#bar graph
X.mean().plot(kind="bar")
plt.xlabel("Genes")
plt.ylabel("Mean Expression")
plt.title("Average Gene Expression")
plt.show()
#scatter plot
plt.scatter(data["BRCA1"], data["TP53"])
plt.xlabel("BRCA1 Expression")
plt.ylabel("TP53 Expression")
plt.title("BRCA1 vs TP53 Expression")
plt.show()
#Histogram
plt.hist(data["TP53"], bins=10)
plt.xlabel("TP53 Expression")
plt.ylabel("Frequency")
plt.title("Distribution of TP53 Expression")
plt.show()
#line graph
plt.plot(X.iloc[0])
plt.xlabel("Genes")
plt.ylabel("Expression")
plt.title("Gene Expression Profile of Sample 1")
plt.xticks(range(len(X.columns)), X.columns, rotation=45)
plt.show()




plt.plot(X.iloc[0])
plt.xlabel("Genes")
plt.ylabel("Expression")
plt.title("Gene Expression Profile of Sample 1")
plt.xticks(range(len(X.columns)), X.columns, rotation=45)
plt.show()