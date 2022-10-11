from copy import deepcopy
from pandas import get_dummies, DataFrame
from typing import List


def modify_space_data(dataset: "DataFrame") -> "DataFrame":
    """
    Function for converting the space dataset into one
    that's usable for numeric inference.
    Works for both the train and test datasets when in
    pandas Dataframe form.
    """

    # DeepCopy the dataframe
    edit_data: "DataFrame" = deepcopy(dataset)

    # Create columns names into which "Cabin" will be split
    cabin_split: "List[str]" = ["Deck", "Num", "Side"]

    # Create list of columns that will be droppped
    to_drop: "List[str]" = ["Cabin", "Num", "PassengerId", "Name"]

    # Create list of columns that have pure numeric data, excluding "Age"
    numerics: "List[str]" = [
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
    ]

    # Split the "Cabin" column into three other columns
    edit_data[cabin_split] = edit_data["Cabin"].str.split(pat="/", expand=True)

    # Drop the irrelevant data
    edit_data = edit_data.drop(labels=to_drop, axis=1)

    # Add "IsZero" indicator features for selected numeric variables
    for i in numerics:
        edit_data[i + "IsZero"] = edit_data[i].map(lambda x: x == 0.0)

    # Create dummy variables for categorical variables in dataset
    # including indicators for "IsNaN"
    edit_data = get_dummies(data=edit_data, drop_first=True, dummy_na=True)

    # Create "IsNaN" indicator variables for all numeric variables (including "Age")
    # Then fill NaN values with zero
    for i in numerics + ["Age"]:
        edit_data[i + "IsNaN"] = edit_data[i].isna()
        edit_data[i] = edit_data[i].fillna(0)

    # Return modified dataset
    return edit_data
