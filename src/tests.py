import classes
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import unittest

from datetime import date, datetime
from global_settings import NBP_URL
from pandas.testing import assert_frame_equal
from parameterized import parameterized
from unittest.mock import ANY, patch
from urllib.error import URLError


CORRECT_URL_EXTENSION = 'USD/2024-09-05/2024-10-06?format=json'
FULL_URL = NBP_URL + CORRECT_URL_EXTENSION
RESPONSE = (
    b'{"table":"C","currency":"dolar ameryka\xc5\x84ski","code":"USD",'
    + b'"rates":[{"no":"173/C/NBP/2024","effectiveDate":"2024-10-01",'
    + b'"bid":4.05,"ask":4.06},'
    + b'{"no":"174/C/NBP/2024","effectiveDate":"2024-10-02",'
    + b'"bid":4.09,"ask":4.10},'
    + b'{"no":"175/C/NBP/2024","effectiveDate":"2024-10-03",'
    + b'"bid":4.08,"ask":4.11}]}'
)
EMPTY_URL_EXTENSION = 'empty_frame'
EMPTY_URL = NBP_URL + EMPTY_URL_EXTENSION
EMPTY_RESPONSE = (
    b'{"table":"C","currency":"dolar ameryka\xc5\x84ski","code":"USD",'
    + b'"rates":[]}'
)


class TestNBPAnalyser(unittest.TestCase):
    def test_initialize_default(self):
        analyser = classes.NBPAnalyser()

        self.assertEqual(analyser.drop_id, False)
        self.assertEqual(analyser.timeout, 30)

    def test_initialize(self):
        analyser = classes.NBPAnalyser(True, 20)

        self.assertEqual(analyser.drop_id, True)
        self.assertEqual(analyser.timeout, 20)
    
    def test_timeout0(self):
        with self.assertRaises(ValueError):
            classes.NBPAnalyser(True, 0)
        
    def test_invalid_timeout(self):
        with self.assertRaises(ValueError):
            classes.NBPAnalyser(True, -5)
        
    def test_drop_id_property(self):
        analyser = classes.NBPAnalyser()
        analyser.drop_id = True

        self.assertEqual(analyser.drop_id, True)
    
    def test_timeout_property(self):
        analyser = classes.NBPAnalyser()
        analyser.timeout = 20

        self.assertEqual(analyser.timeout, 20)
    
    def test_timeout_property_error0(self):
        analyser = classes.NBPAnalyser()

        with self.assertRaises(ValueError):
            analyser.timeout = 0
    
    def test_timeout_property_error(self):
        analyser = classes.NBPAnalyser()

        with self.assertRaises(ValueError):
            analyser.timeout = -5
    
    @parameterized.expand([
        ('', ''), 
        ('USD', 'USD'), 
        ('usd', 'USD'),
        ('U \tSD', 'USD'),
        ('U\nSD', 'USD'),
        (' USD \n ', 'USD'),
        ('EUR', 'EUR'),
        ('eur', 'EUR'),
        ('E \tUR', 'EUR'),
        ('E\nUR', 'EUR'),
        (' EUR \n ', 'EUR'),
    ])
    def test_format_code(self, input, expected):
        self.assertEqual(classes.NBPAnalyser.format_code(input), expected)
    
    def test_get_extension(self):
        start_date = date(2024, 9, 5)
        end_date = date(2024, 10, 6)
        currency = ' uSd '
        expected = CORRECT_URL_EXTENSION

        self.assertEqual(
            classes.NBPAnalyser.get_extension(
                start_date,
                end_date,
                currency,
            ),
            expected,
        )
    
    def test_invalid_extension_data(self):
        start_date = date(2024, 10, 6)
        end_date = date(2024, 9, 5)
        currency = ' uSd '

        with self.assertRaises(ValueError):
            classes.NBPAnalyser.get_extension(
                start_date,
                end_date,
                currency,
            )

    def test_wrong_dataframe_summary(self):
        df = pd.DataFrame()
        df['effectiveDate'] = [
            datetime(2024, 10, 1),
            datetime(2024, 10, 2),
            datetime(2024, 10, 3),
        ]
        df['bid'] = [
            4.05,
            4.09,
            4.08,
        ]
        df['ask'] = [
            4.06,
            4.10,
            4.11,
        ]

        with self.assertRaises(ValueError):
            classes.NBPAnalyser.get_summary(df)

    def test_empty_dataframe_summary(self):
        df = pd.DataFrame()

        with self.assertRaises(ValueError):
            classes.NBPAnalyser.get_summary(df)
    
    def test_get_summary(self):
        df = pd.DataFrame()
        df['effectiveDate'] = [
            datetime(2024, 10, 1),
            datetime(2024, 10, 2),
            datetime(2024, 10, 3),
        ]
        df['bid'] = [
            4.05,
            4.09,
            4.08,
        ]
        df['ask'] = [
            4.06,
            4.10,
            4.11,
        ]
        df['spread'] = [
            0.01,
            0.01,
            0.03,
        ]
        expected = pd.DataFrame()
        expected['min'] = [
            4.06,
            4.05,
            0.01,
        ]
        expected['mean'] = [
            4.09,
            4.0733333333,
            0.0166666666,
        ]
        expected['max'] = [
            4.11,
            4.09,
            0.03,
        ]
        expected.index = ['ask', 'bid', 'spread']

        result = classes.NBPAnalyser.get_summary(df)

        assert_frame_equal(result, expected)

@patch.object(classes.sns, 'histplot', wraps=sns.histplot)
class TestHistograms(unittest.TestCase):
    def test_wrong_dataframe(self, hist_mock):
        df = pd.DataFrame()
        df['effectiveDate'] = [
            datetime(2024, 10, 1),
            datetime(2024, 10, 2),
            datetime(2024, 10, 3),
        ]
        df['bid'] = [
            4.05,
            4.09,
            4.08,
        ]
        df['ask'] = [
            4.06,
            4.10,
            4.11,
        ]

        with self.assertRaises(ValueError):
            classes.NBPAnalyser.draw_histograms(df)

    def test_empty_dataframe(self, hist_mock):
        df = pd.DataFrame()

        with self.assertRaises(ValueError):
            classes.NBPAnalyser.draw_histograms(df)
    
    def test_dataframe(self, hist_mock):
        df = pd.DataFrame()
        df['effectiveDate'] = [
            datetime(2024, 10, 1),
            datetime(2024, 10, 2),
            datetime(2024, 10, 3),
        ]
        df['bid'] = [
            4.05,
            4.09,
            4.08,
        ]
        df['ask'] = [
            4.06,
            4.10,
            4.11,
        ]
        df['spread'] = [
            0.01,
            0.01,
            0.03,
        ]
        melted_df = pd.DataFrame()
        melted_df['effectiveDate'] = (
            datetime(2024, 10, 1),
            datetime(2024, 10, 2),
            datetime(2024, 10, 3),
            datetime(2024, 10, 1),
            datetime(2024, 10, 2),
            datetime(2024, 10, 3),
        )
        melted_df['variable'] = (
            'bid',
            'bid',
            'bid',
            'ask',
            'ask',
            'ask',
        )
        melted_df['value'] = (
            4.05,
            4.09,
            4.08,
            4.06,
            4.10,
            4.11,
        )

        classes.NBPAnalyser.draw_histograms(df)
        args = hist_mock.call_args.args

        assert_frame_equal(melted_df, args[0])

    def test_return(self, hist_mock):
        df = pd.DataFrame()
        df['effectiveDate'] = [
            datetime(2024, 10, 1),
            datetime(2024, 10, 2),
            datetime(2024, 10, 3),
        ]
        df['bid'] = [
            4.05,
            4.09,
            4.08,
        ]
        df['ask'] = [
            4.06,
            4.10,
            4.11,
        ]
        df['spread'] = [
            0.01,
            0.01,
            0.03,
        ]

        fig = classes.NBPAnalyser.draw_histograms(df)

        self.assertIsInstance(fig, plt.Figure)

class TestSeries(unittest.TestCase):
    def test_wrong_dataframe(self):
        df = pd.DataFrame()
        df['effectiveDate'] = [
            datetime(2024, 10, 1),
            datetime(2024, 10, 2),
            datetime(2024, 10, 3),
        ]
        df['bid'] = [
            4.05,
            4.09,
            4.08,
        ]
        df['ask'] = [
            4.06,
            4.10,
            4.11,
        ]

        with self.assertRaises(ValueError):
            classes.NBPAnalyser.draw_time_series(df)

    def test_empty_dataframe(self):
        df = pd.DataFrame()

        with self.assertRaises(ValueError):
            classes.NBPAnalyser.draw_time_series(df)
    
    def test_return(self):
        df = pd.DataFrame()
        df['effectiveDate'] = [
            datetime(2024, 10, 1),
            datetime(2024, 10, 2),
            datetime(2024, 10, 3),
        ]
        df['bid'] = [
            4.05,
            4.09,
            4.08,
        ]
        df['ask'] = [
            4.06,
            4.10,
            4.11,
        ]
        df['spread'] = [
            0.01,
            0.01,
            0.03,
        ]

        fig = classes.NBPAnalyser.draw_time_series(df)

        self.assertIsInstance(fig, go.Figure)

class FakeHTTPResponse:
    def read(self):
        return RESPONSE

class EmptyFakeHTTPResponse:
    def read(self):
        return EMPTY_RESPONSE

def urlopen_side_effect(url: str, timeout: float):
    if url == FULL_URL:
        return FakeHTTPResponse()
    if url == EMPTY_URL:
        return EmptyFakeHTTPResponse()
    raise URLError('Failed to load data')

@patch('classes.urlopen')
class TestDownloadData(unittest.TestCase):
    def test_regular(self, urlopen_mock):
        urlopen_mock.side_effect = urlopen_side_effect
        analyser = classes.NBPAnalyser()
        expected = pd.DataFrame()
        expected['effectiveDate'] = [
            datetime(2024, 10, 1),
            datetime(2024, 10, 2),
            datetime(2024, 10, 3),
        ]
        expected['bid'] = [
            4.05,
            4.09,
            4.08,
        ]
        expected['ask'] = [
            4.06,
            4.10,
            4.11,
        ]
        expected['spread'] = [
            0.01,
            0.01,
            0.03,
        ]
        expected.index = [
            '173/C/NBP/2024',
            '174/C/NBP/2024',
            '175/C/NBP/2024',
        ]

        result = analyser.download_data(CORRECT_URL_EXTENSION)

        assert_frame_equal(result, expected)
    
    def test_drop_id(self, urlopen_mock):
        urlopen_mock.side_effect = urlopen_side_effect
        analyser = classes.NBPAnalyser(drop_id=True)
        expected = pd.DataFrame()
        expected['effectiveDate'] = [
            datetime(2024, 10, 1),
            datetime(2024, 10, 2),
            datetime(2024, 10, 3),
        ]
        expected['bid'] = [
            4.05,
            4.09,
            4.08,
        ]
        expected['ask'] = [
            4.06,
            4.10,
            4.11,
        ]
        expected['spread'] = [
            0.01,
            0.01,
            0.03,
        ]

        result = analyser.download_data(CORRECT_URL_EXTENSION)

        assert_frame_equal(result, expected)
    
    def test_url_error(self, urlopen_mock):
        urlopen_mock.side_effect = urlopen_side_effect
        analyser = classes.NBPAnalyser()

        with self.assertRaises(URLError):
            analyser.download_data('blabla')
    
    def test_empty_data(self, urlopen_mock):
        urlopen_mock.side_effect = urlopen_side_effect
        analyser = classes.NBPAnalyser()

        with self.assertRaises(URLError):
            analyser.download_data(EMPTY_URL_EXTENSION)
    
    def test_passed_arguments(self, urlopen_mock):
        urlopen_mock.side_effect = urlopen_side_effect
        analyser = classes.NBPAnalyser(timeout=20)

        analyser.download_data(CORRECT_URL_EXTENSION)
        args = urlopen_mock.call_args.args
        kwargs = urlopen_mock.call_args.kwargs

        self.assertEqual(args[0], FULL_URL)
        self.assertAlmostEqual(kwargs['timeout'], 20)
