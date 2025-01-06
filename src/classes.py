import datetime as dt
import json
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from abc import ABC, abstractmethod
from global_settings import NBP_URL
from io import BytesIO
from plotly.subplots import make_subplots
from urllib.request import urlopen


class DataDownloader(ABC):
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    @abstractmethod
    def download_data(self, url_extension: str):
        pass

class NBPAnalyser(DataDownloader):
    def __init__(self, drop_id=False, timeout=30):
        super().__init__(NBP_URL)
        self._drop_id = drop_id
        self._timeout = timeout
    
    @property
    def drop_id(self):
        return self._drop_id

    @drop_id.setter
    def drop_id(self, value: bool):
        self._drop_id = value
    
    @property
    def timeout(self):
        return self._timeout

    @drop_id.setter
    def drop_id(self, value: bool):
        if value <= 0:
            raise ValueError('Timeout must be positive')
        self._drop_id = value
    
    @staticmethod
    def _melt_data(data: pd.DataFrame) -> pd.DataFrame:
        value_cols = [col for col in data.columns if col != 'effectiveDate']
        melted_data = data.melt(
            id_vars=['effectiveDate'],
            value_vars=value_cols,
        )

        return melted_data

    @staticmethod
    def format_code(code: str) -> str:
        return code.strip().replace(' ', '').upper()
    
    @staticmethod
    def get_extension(
        start_date: dt.date,
        end_date: dt.date,
        currency: str
    ) -> str:
        if end_date < start_date:
            raise ValueError('end_date must be after start_date')
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        return f'{currency}/{start_date_str}/{end_date_str}?format=json'
    
    @staticmethod
    def get_summary(data: pd.DataFrame) -> pd.DataFrame:
        melted_data = NBPAnalyser._melt_data(data)

        operations = ['min', 'mean', 'max']
        result = melted_data.groupby('variable').agg({
            'value' : ['min', 'mean', 'max']
        })
        result.index.name = None
        result.columns = operations

        return result
    
    @staticmethod
    def draw_histograms(data: pd.DataFrame, width=7, height=7) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(width, height))
        data = data.drop('spread', axis=1)
        melted_data = NBPAnalyser._melt_data(data)
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

        ax.set(xlabel=None)
        ax.get_legend().set_title(None)

        return fig
    
    @staticmethod
    def draw_time_series(data: pd.DataFrame) -> go.Figure:
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
        data['effectiveDate'] = pd.to_datetime(data['effectiveDate'])

        if self._drop_id:
            data = data.drop('no', axis=1)
        else:
            data = data.set_index('no')
            data.index.name = None
        
        data['spread'] = data['ask'] - data['bid']
        
        return data
    
    def download_data(self, url_extension: str) -> pd.DataFrame:
        resp = urlopen(self.base_url + url_extension, timeout=self._timeout)
        json_result = resp.read()
        json_data = json.load((BytesIO(json_result)))
        result = pd.DataFrame(json_data['rates'])
        
        return self._process_data(result)
