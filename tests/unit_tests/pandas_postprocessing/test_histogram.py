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
from superset.utils.pandas_postprocessing import histogram

data = DataFrame(
    {
        "group": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "A"],
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
)

bins = 5


def test_histogram_no_groupby():
    data_with_no_groupings = DataFrame(
        {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    )
    result = histogram(data_with_no_groupings, "a", [], bins)
    assert result.shape == (1, bins)
    assert result.columns.tolist() == [
        "1.0 - 2.8",
        "2.8 - 4.6",
        "4.6 - 6.4",
        "6.4 - 8.2",
        "8.2 - 10.0",
    ]
    assert result.values.tolist() == [[2, 2, 2, 2, 2]]


def test_histogram_with_groupby():
    result = histogram(data, "a", ["group"], bins)
    assert result.shape == (2, bins + 1)
    assert result.columns.tolist() == [
        "group",
        "1.0 - 2.8",
        "2.8 - 4.6",
        "4.6 - 6.4",
        "6.4 - 8.2",
        "8.2 - 10.0",
    ]
    assert result.values.tolist() == [["A", 2, 0, 2, 0, 2], ["B", 0, 2, 0, 2, 0]]


def test_histogram_with_groupby_and_normalize():
    result = histogram(data, "a", ["group"], bins, normalize=True)
    assert result.shape == (2, bins + 1)
    assert result.columns.tolist() == [
        "group",
        "1.0 - 2.8",
        "2.8 - 4.6",
        "4.6 - 6.4",
        "6.4 - 8.2",
        "8.2 - 10.0",
    ]
    assert result.values.tolist() == [
        ["A", 0.2, 0.0, 0.2, 0.0, 0.2],
        ["B", 0.0, 0.2, 0.0, 0.2, 0.0],
    ]


def test_histogram_with_groupby_and_cumulative():
    result = histogram(data, "a", ["group"], bins, cumulative=True)
    assert result.shape == (2, bins + 1)
    assert result.columns.tolist() == [
        "group",
        "1.0 - 2.8",
        "2.8 - 4.6",
        "4.6 - 6.4",
        "6.4 - 8.2",
        "8.2 - 10.0",
    ]
    assert result.values.tolist() == [["A", 2, 2, 4, 4, 6], ["B", 0, 2, 2, 4, 4]]


def test_histogram_with_groupby_and_cumulative_and_normalize():
    result = histogram(data, "a", ["group"], bins, cumulative=True, normalize=True)
    assert result.shape == (2, bins + 1)
    assert result.columns.tolist() == [
        "group",
        "1.0 - 2.8",
        "2.8 - 4.6",
        "4.6 - 6.4",
        "6.4 - 8.2",
        "8.2 - 10.0",
    ]
    # The fixture holds 10 observations (6 in A, 4 in B). Cumulative counts are
    # A: [2, 2, 4, 4, 6] and B: [0, 2, 2, 4, 4]; normalizing divides each by the
    # total observation count (10), so each group ends at its share of the data
    # (0.6 + 0.4 = 1.0) rather than dividing by the sum of the cumulative counts.
    assert result.values.tolist() == [
        ["A", 0.2, 0.2, 0.4, 0.4, 0.6],
        ["B", 0.0, 0.2, 0.2, 0.4, 0.4],
    ]


def test_histogram_cumulative_normalized_no_groupby():
    result = histogram(
        DataFrame({"value": [1, 2, 3, 4]}),
        "value",
        [],
        2,
        cumulative=True,
        normalize=True,
    )
    assert result.shape == (1, 2)
    assert result.columns.tolist() == ["1.0 - 2.5", "2.5 - 4.0"]
    # raw counts [2, 2] -> cumulative [2, 4] -> divided by 4 observations
    assert result.values.tolist()[0] == pytest.approx([0.5, 1.0])

    result = histogram(
        DataFrame({"value": [1, 2, 3, 4, 5, 6]}),
        "value",
        [],
        3,
        cumulative=True,
        normalize=True,
    )
    assert result.shape == (1, 3)
    # raw counts [2, 2, 2] -> cumulative [2, 4, 6] -> divided by 6 observations
    assert result.values.tolist()[0] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert result.values[0, -1] == pytest.approx(1.0)


def test_histogram_cumulative_normalized_groupby():
    unequal_groups = DataFrame(
        {
            "group": ["X"] * 7 + ["Y"] * 3,
            "value": [1, 2, 3, 4, 5, 6, 7, 2, 4, 6],
        }
    )
    result = histogram(
        unequal_groups, "value", ["group"], 2, cumulative=True, normalize=True
    )
    assert result.shape == (2, 3)
    assert result.columns.tolist() == ["group", "1.0 - 4.0", "4.0 - 7.0"]
    assert result["group"].tolist() == ["X", "Y"]
    # 10 observations in total. Bin edges are [1, 4, 7], so
    # X: raw [3, 4] -> cumulative [3, 7]; Y: raw [1, 2] -> cumulative [1, 3].
    # Dividing by 10 gives each group's share of all observations.
    values = result.drop(columns="group").values
    assert values[0].tolist() == pytest.approx([0.3, 0.7])
    assert values[1].tolist() == pytest.approx([0.1, 0.3])
    assert values[:, -1].sum() == pytest.approx(1.0)


def test_histogram_with_non_numeric_column():
    try:
        histogram(data, "group", None, bins)
    except ValueError as e:
        assert str(e) == "Column 'group' contains non-numeric values"  # noqa: PT017


def test_histogram_with_some_non_numeric_values():
    data_with_non_numeric = DataFrame(
        {
            "group": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "A"],
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, "10"],
            "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, "10"],
        }
    )
    try:
        histogram(data_with_non_numeric, "a", ["group"], bins)
    except ValueError as e:
        assert str(e) == "Column 'group' contains non-numeric values"  # noqa: PT017


def test_histogram_with_groupby_and_some_null_values():
    data_with_groupby_and_some_nulls = DataFrame(
        {
            "group": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "A"],
            "a": [1, 2, 3, 4, 5, None, 7, 8, 9, 10],
            "b": [1, 2, 3, 4, 5, None, 7, 8, 9, 10],
        }
    )

    result = histogram(data_with_groupby_and_some_nulls, "a", ["group"], bins)
    assert result.shape == (2, bins + 1)
    assert result.columns.tolist() == [
        "group",
        "1.0 - 2.8",
        "2.8 - 4.6",
        "4.6 - 6.4",
        "6.4 - 8.2",
        "8.2 - 10.0",
    ]
    assert result.values.tolist() == [["A", 2, 0, 1, 0, 2], ["B", 0, 2, 0, 2, 0]]


def test_histogram_with_no_groupby_and_some_null_values():
    data_with_no_groupby_and_some_nulls = DataFrame(
        {
            "a": [1, 2, 3, 4, 5, None, 7, 8, 9, 10],
            "b": [1, 2, 3, 4, 5, None, 7, 8, 9, 10],
        }
    )

    result = histogram(data_with_no_groupby_and_some_nulls, "a", [], bins)
    assert result.shape == (1, bins)
    assert result.columns.tolist() == [
        "1.0 - 2.8",
        "2.8 - 4.6",
        "4.6 - 6.4",
        "6.4 - 8.2",
        "8.2 - 10.0",
    ]
    assert result.values.tolist() == [[2, 2, 1, 2, 2]]


def test_histogram_with_groupby_and_all_null_values():
    data_with_groupby_and_all_nulls = DataFrame(
        {
            "group": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "A"],
            "a": [None, None, None, None, None, None, None, None, None, None],
            "b": [None, None, None, None, None, None, None, None, None, None],
        }
    )

    result = histogram(data_with_groupby_and_all_nulls, "a", ["group"], bins)
    assert result.empty


def test_histogram_with_no_groupby_and_all_null_values():
    data_with_no_groupby_and_all_nulls = DataFrame(
        {
            "a": [None, None, None, None, None, None, None, None, None, None],
            "b": [None, None, None, None, None, None, None, None, None, None],
        }
    )

    result = histogram(data_with_no_groupby_and_all_nulls, "a", [], bins)
    assert result.empty


def test_histogram_rejects_unbounded_bins():
    """
    ``bins`` comes from the unvalidated post-processing options dict; an
    unbounded value would make numpy allocate a bin-edge array of that size
    (bins=2e9 attempts ~16 GB). Non-integer values such as "auto" are also
    rejected, since data-driven bin estimation is likewise unbounded.
    """
    for bad_bins in (0, -1, 2_000_000_000, "auto", None):
        with pytest.raises(InvalidPostProcessingError):
            histogram(data, "a", [], bad_bins)


def test_histogram_rejects_bool_bins():
    """
    ``bool`` is a subclass of ``int`` in Python, so ``isinstance(bins, int)``
    alone accepts ``True``/``False``. Both must still be rejected since
    neither is a valid bin count.
    """
    for bad_bins in (True, False):
        with pytest.raises(InvalidPostProcessingError):
            histogram(data, "a", [], bad_bins)
