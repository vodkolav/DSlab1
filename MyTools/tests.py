

from MyTools.Viz import summary, save_df_as_pretty_html
import pandas as pd 

df = pd.read_csv("Capstone/dataall_summaries.csv")


summ = summary(df, hist_fmt='base64')

save_df_as_pretty_html(summ, filename="summary_test.html", index=False)