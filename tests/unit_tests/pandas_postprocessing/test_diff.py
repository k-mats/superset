# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest
from pandas import DataFrame

from superset.exceptions import InvalidPostProcessingError
from superset.utils.pandas_postprocessing import diff
from tests.unit_tests.fixtures.dataframes import timeseries_df, timeseries_df2
from tests.unit_tests.pandas_postprocessing.utils import series_to_list


def test_diff():
    # overwrite column
    post_df = diff(df=timeseries_df, columns={"y": "y"})
    assert post_df.columns.tolist() == ["label", "y"]
    assert series_to_list(post_df["y"]) == [None, 1.0, 1.0, 1.0]

    # add column
    post_df = diff(df=timeseries_df, columns={"y": "y1"})
    assert post_df.columns.tolist() == ["label", "y", "y1"]
    assert series_to_list(post_df["y"]) == [1.0, 2.0, 3.0, 4.0]
    assert series_to_list(post_df["y1"]) == [None, 1.0, 1.0, 1.0]

    # look ahead
    post_df = diff(df=timeseries_df, columns={"y": "y1"}, periods=-1)
    assert series_to_list(post_df["y1"]) == [-1.0, -1.0, -1.0, None]

    # invalid column reference
    with pytest.raises(InvalidPostProcessingError):
        diff(
            df=timeseries_df,
            columns={"abc": "abc"},
        )

    # diff by columns
    post_df = diff(df=timeseries_df2, columns={"y": "y", "z": "z"}, axis=1)
    assert post_df.columns.tolist() == ["label", "y", "z"]
    assert series_to_list(post_df["z"]) == [0.0, 2.0, 8.0, 6.0]


def test_diff_mixed_replace_and_append():
    df = DataFrame({"sales": [10, 20, 30], "orders": [1, 2, 3]})
    original_df = df.copy()

    post_df = diff(df=df, columns={"sales": "sales", "orders": "orders_diff"})

    assert post_df.columns.tolist() == ["sales", "orders", "orders_diff"]
    assert series_to_list(post_df["sales"]) == [None, 10.0, 10.0]
    assert post_df["orders"].tolist() == [1, 2, 3]
    assert series_to_list(post_df["orders_diff"]) == [None, 1.0, 1.0]
    assert df.equals(original_df)
