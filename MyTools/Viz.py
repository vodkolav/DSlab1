
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64


def create_histogram(data_series):

    if np.issubdtype(data_series.dtype, np.number):
        pass
    elif np.issubdtype(data_series.dtype, np.datetime64):
        data_series = data_series.astype(np.int64) // 10**9  # Convert to seconds since epoch
    elif np.issubdtype(data_series.dtype, np.bool):
        data_series = data_series.astype(int)
    else:
        data_series = data_series.str.len()
        
    # Create a new plot
    fig, ax = plt.subplots(figsize=(1.0, 0.5), dpi=200)        # Plot the histogram
    nbins = min(10, data_series.nunique())
    ax.hist(data_series, bins=nbins, edgecolor='none', color='#4CAF50')
    
    # Hide axes and labels for a cleaner "sparkline" look
    #ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    # Use an in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
    
    # Close the plot to free memory
    plt.close(fig)
    
    return  buf.getvalue()



def create_base64_histogram(data_series):
    imgdata = create_histogram(data_series)
    # Encode the image to base64
    data_uri = base64.b64encode(imgdata).decode('utf-8')
    
    # Create an HTML image tag
    return f'<img src="data:image/png;base64,{data_uri}">'


def create_png_histogram(data_series):
    imgdata = create_histogram(data_series)
    return imgdata



def summary(df, hist_fmt = 'base64'):

    if hist_fmt == 'base64':
        hists = df.apply(lambda row: create_base64_histogram(row), axis=0)
    elif hist_fmt == 'png':
        hists = df.apply(lambda row: create_histogram(row), axis=0)

    #hists = 'Histogram' * df.shape[0]

    nonnans = df.shape[0] - df.isna().sum()
    nonnansPrc = (nonnans / df.shape[0] * 100).apply("{0:.2f}%".format)
    sam1 = df.sample(1, random_state=42).squeeze()
    sam2 = df.sample(1, random_state=495).squeeze()
    res = pd.DataFrame([sam1.index, df.dtypes.astype(str), nonnans,
                        nonnansPrc, df.nunique(), hists, sam1, sam2]).transpose()
    res.columns = ["Column", "data type", "non-null values", 
                   "non-null values %", "unique values","Hist", "example1", "example2"]
    


    res.sort_values([ "non-null values","unique values"],ascending=False, inplace=True)
    # let's abuse python's very lascivious OOP system
    res._repr_html_ = lambda: res.to_html(escape=False)
    return res



def summary_hists(summary_df, ):

    hists_data = df.apply(lambda row: create_png_histogram(row), axis=0)

    link = f'![Histogram]({os.path.join(IMAGE_DIR, filename)})'


def save_df_as_pretty_html(df, filename="output.html", index=True):
    pd.set_option("display.max_colwidth", None)
    # Convert newlines to <br> for HTML
    df_html_ready = df.copy()
    for col in df_html_ready.columns:
        df_html_ready[col] = df_html_ready[col] \
                                .astype(str)\
                                .str.replace('\n', '<br>', regex=False)

    # Generate styled HTML
    html = df_html_ready.to_html(
        escape=False,  # Needed to render <br>
        index=index,
        border=0,
        classes="styled-table"
    )

    # Add CSS styling
    style = """
    <style>
    .styled-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 16px;
        font-family: Arial, sans-serif;
        width: 100%;
        table-layout: auto; /* ✅ Let browser fit naturally */
    }
    .styled-table th, .styled-table td {
        border: 1px solid #dddddd;
        padding: 10px;
        vertical-align: top;
        text-align: left;
        overflow-wrap: break-word; /* ✅ Break inside words */
        white-space: pre-wrap; /* ✅ Honor \\n linebreaks */
    }
    .styled-table td {
        max-width: 600px; /* ✅ Avoid huge dream fields expanding table */
    }
    .styled-table th {
        background-color: #f2f2f2;
    }
    </style>
    """

    # Write full HTML document
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html><html><head>{style}</head><body>{html}</body></html>")

    print(f"✅ HTML table saved to: {filename}")





