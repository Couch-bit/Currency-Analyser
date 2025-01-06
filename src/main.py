import classes
import seaborn as sns
import streamlit as st

from datetime import date
from dateutil.relativedelta import relativedelta
from io import BytesIO


# configures general page layout
st.set_page_config(layout='wide', page_title="Currency Dashboard")
st.title('_:blue[Currency] Dashboard_')
settings, result = st.columns([0.2, 0.8])
# configures seaborn theme
sns.set_theme(rc={
    'figure.facecolor': '#0e1117',
    'axes.facecolor': 'black',
    'text.color': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.labelcolor': 'white',
})

# makes sure 'data_ready' is defined is session_state
if 'data_ready' not in st.session_state:
    st.session_state['data_ready'] = False

# makes sure 'analyser' is defined is session_state
if 'analyser' not in st.session_state:
    st.session_state['analyser'] = classes.NBPAnalyser()

with settings:
    # allows for control of how much data to download
    month_num = st.slider(
        'Number of months',
        min_value=1,
        max_value=12,
        value=6,
    ) 

    # data is gathered from NBP API
    currency_code = (
        st.text_input('Enter currency code:')
    )

    if st.button('Download Data'):
        # don't show anything if last data load failed
        st.session_state['data_ready'] = False

        # formats code to remove spaces and make it upper case
        formatted_code = st.session_state['analyser'].format_code(
            currency_code
        )
        
        try:
            # saves which currency the data is for
            st.session_state['currency'] = formatted_code
            # gets url_extension
            current_date = date.today()
            url_extension = st.session_state['analyser'].get_extension(
                current_date - relativedelta(months=month_num),
                current_date,
                formatted_code,
            )
            # tries to get data, if successful updates session state
            st.session_state['df'] = (
                st.session_state['analyser'].download_data(url_extension)
            )
            # data loaded successfully
            st.session_state['data_ready'] = True
        except Exception:
            st.write('Unexpected error occurred')
    
    # allows for control of data shown in preview
    row_num = st.slider(
        'Number of rows',
        min_value=5,
        max_value=20,
    )

with result:
    if st.session_state['data_ready']:
        # sets header as the name of the stock
        st.header(f'Analysis for Currency: {st.session_state['currency']}')
        
        data_tab, chart_tab = st.tabs(['🗂️ Data', '📈 Plots'])
        with data_tab:
            # gets DataFrame to be displayed
            df_display = st.session_state['df'].copy()
            df_display = df_display.set_index('effectiveDate')
            df_display.index = df_display.index.date
            df_display.index.name = None
            # displays a preview of the data
            st.subheader('Raw Data')
            st.write(df_display.tail(row_num))

            # displays data summary
            st.subheader('Summary')
            st.write(
                st.session_state['analyser'].get_summary(
                    st.session_state['df']
                )
            )

        with chart_tab:
            st.subheader('Histogram of Rates')
            # displays histograms of bid and ask rates
            hist_fig = st.session_state['analyser'].draw_histograms(
                st.session_state['df']
            )
            buf = BytesIO()
            hist_fig.savefig(buf, format='png')
            st.image(buf)

            # displays time series of bid and ask rates
            st.subheader('Time Series of Rates')
            series_fig = st.session_state['analyser'].draw_time_series(
                st.session_state['df']
            )
            st.plotly_chart(series_fig)
