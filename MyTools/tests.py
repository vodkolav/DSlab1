
from MyTools.DataIO import jload, jsave

from MyTools.Viz import summary, save_df_as_pretty_html, j_summary
import pandas as pd 

df = pd.read_csv("Capstone/dataall_summaries.csv")


# summ = summary(df)

# save_df_as_pretty_html(summ, filename="summary_test.html", index=False)
data = jload("Capstone/articles.json")

uhm = j_summary(data)

jsave(uhm, "Capstone/articles_summary.json")