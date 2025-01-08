import datetime as dt
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from abc import ABC, abstractmethod
from global_settings import DEFAULT_TIMEOUT, NBP_URL
from io import BytesIO
from plotly.subplots import make_subplots
from urllib.error import URLError
from urllib.request import urlopen


class DataDownloader(ABC):
    """Abstract class for objects which download data"""
    def __init__(self, base_url: str):
        self._base_url = base_url
    
    @abstractmethod
    def download_data(self, url_extension: str):
        pass

class NBPAnalyser(DataDownloader):
    """Class for objects which download data
    from NBP API"""
    def __init__(self, drop_id=False, timeout=DEFAULT_TIMEOUT):
        if timeout <= 0:
            raise ValueError('timeout must be positive')
        super().__init__(NBP_URL)
        self._drop_id = drop_id
        self._timeout = timeout
    
    @property
    def drop_id(self) -> bool:
        """Whether to drop the NBP ID or not"""
        return self._drop_id

    @drop_id.setter
    def drop_id(self, value: bool):
        self._drop_id = value
    
    @property
    def timeout(self) -> float:
        """Time before request times out, must be positive"""
        return self._timeout

    @timeout.setter
    def timeout(self, value: int):
        if value <= 0:
            raise ValueError('timeout must be positive')
        self._timeout = value
    
    @staticmethod
    def _melt_data(data: pd.DataFrame) -> pd.DataFrame:
        # remove date from value_vars
        value_cols = [col for col in data.columns if col != 'effectiveDate']
        melted_data = data.melt(
            id_vars=['effectiveDate'],
            value_vars=value_cols,
        )

        return melted_data
    
    @staticmethod
    def _check_frame(data: pd.DataFrame) -> bool:
        expected_columns = np.array(['effectiveDate', 'bid', 'ask', 'spread'])
        
        # DataFrame is incorrect if it has invalid columns
        # or no rows
        return (
            len(data.columns) == len(expected_columns)
            and (data.columns.to_numpy() == expected_columns).all()
            and len(data) > 0
        )

    @staticmethod
    def format_code(code: str) -> str:
        """Formats given currency code by removing whitespace
        and making it upper case
        """
        return ''.join(code.split()).upper()
    
    @staticmethod
    def get_extension(
        start_date: dt.date,
        end_date: dt.date,
        currency: str
    ) -> str:
        """Given start and end date combined with currency code creates
        extension necessary for the API call
        """
        if end_date < start_date:
            raise ValueError('end_date must be after start_date')
        
        # gets necesary parts for URL
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        currency = NBPAnalyser.format_code(currency)

        return f'{currency}/{start_date_str}/{end_date_str}?format=json'
    
    @staticmethod
    def get_summary(data: pd.DataFrame) -> pd.DataFrame:
        """Creates summary (min, mean, max) for 
        numeric columns of downloaded data"""
        if not NBPAnalyser._check_frame(data):
            raise ValueError('incorrect format of DataFrame')
        
        melted_data = NBPAnalyser._melt_data(data)

        # defines necessary operations
        operations = ['min', 'mean', 'max']
        result = melted_data.groupby('variable').agg({
            'value' : ['min', 'mean', 'max']
        })

        # cleans up index and columns
        result.index.name = None
        result.columns = operations

        return result
    
    @staticmethod
    def draw_histograms(data: pd.DataFrame) -> plt.Figure:
        """Draws histograms with kernel density estimates
        of bid and ask rates on one seaborn plot"""
        if not NBPAnalyser._check_frame(data):
            raise ValueError('incorrect format of DataFrame')
        
        # 5x5 is a good base size for web apps
        fig, ax = plt.subplots(figsize=(5, 5))
        data = data.drop('spread', axis=1)
        melted_data = NBPAnalyser._melt_data(data)

        # plots histograms
        sns.histplot(
            melted_data,
            x='value',
            hue='variable',
            stat="density",
            element='step',
            common_norm=False,
            kde=True,
            ax=ax,
        )

        # removes unnecessary labels
        ax.set(xlabel=None)
        ax.get_legend().set_title(None)

        return fig
    
    @staticmethod
    def draw_time_series(data: pd.DataFrame) -> go.Figure:
        """Draws time series of bid and ask rates on one plotly plot
        designed for a dark background"""
        if not NBPAnalyser._check_frame(data):
            raise ValueError('incorrect format of DataFrame')
        
        fig = make_subplots()

        # adds lineplot of bid rates
        fig.add_trace(
            go.Scatter(
                x=data['effectiveDate'],
                y=data['bid'],
                line=dict(color='lightblue', width=1),
                name=f'bid',
            )
        )

        # adds lineplot of ask rates
        fig.add_trace(
            go.Scatter(
                x=data['effectiveDate'],
                y=data['ask'],
                line=dict(color='orange', width=1),
                name=f'ask',
            )
        )

        # configures layout
        layout = go.Layout(
            plot_bgcolor='black',
            font_color='white',
            font_size=20,
            xaxis=dict(
                rangeslider=dict(
                    visible=False
                )
            ),
        )
        fig.update_layout(layout)

        return fig

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        # makes sure the date is interpreted as datetime
        data['effectiveDate'] = pd.to_datetime(data['effectiveDate'])

        # removes NBP ID or moves it to index
        if self._drop_id:
            data = data.drop('no', axis=1)
        else:
            data = data.set_index('no')
            data.index.name = None
        
        # calculates spread of exchange rates
        data['spread'] = data['ask'] - data['bid']
        
        return data
    
    def download_data(self, url_extension: str) -> pd.DataFrame:
        """Downloads data from NBP API given url_extension
        created by get_extension or manually, it has the format
        <code>/<start_date>/<end_date>?format=json (dates are
        in the %Y-%m-%d format)"""
        # downloads data in json format
        resp = urlopen(self._base_url + url_extension, timeout=self._timeout)
        json_result = resp.read()
        json_data = json.load((BytesIO(json_result)))

        # converts data to DataFrame
        result = pd.DataFrame(json_data['rates'])

        # throws an exception if no data was downloaded
        if len(result) == 0:
            raise URLError('No data found')
        
        return self._process_data(result)
