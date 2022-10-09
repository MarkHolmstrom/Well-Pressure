#!/usr/bin/env python3

import pandas as pd
import numpy as np


def get_pressure_test_oil_ab(excluded_df, well_list_df):
    """Finds wells in Pressure test OIL AB.csv that have records in a well list,
         but only if they do not already have records in a given dataset.
         excluded_df is the Pandas dataframe that contains wells which should not be added (All_Pp_Grad_Data.csv).
         well_list_df is the Pandas dataframe that contains the well list (Well_List.xlsx)."""
    pressure_test_oil_ab_df = pd.read_csv("Pressure test OIL AB.csv")
    merged_oil_ab_well_list_df = pd.merge(well_list_df, pressure_test_oil_ab_df, how="inner",
                                          left_on="CPA Pretty Well ID", right_on="Unique Well ID")
    # wells from AB oil pressure tests that are in the well list

    pool_prefixes = ("DOIG", "MONT")
    merged_oil_ab_well_list_df = merged_oil_ab_well_list_df.loc[
        merged_oil_ab_well_list_df["Pool Name"].str.startswith(pool_prefixes) |  # | instead of or needed
        merged_oil_ab_well_list_df["Pool Name"].isnull()]
    # selecting wells that are in a pool that begins with a given prefix
    # or do not have a value (redundant in practice, all get eliminated later)

    oil_ab_append_df = pd.merge(excluded_df, merged_oil_ab_well_list_df, how="outer", indicator=True,
                                left_on="Well ID", right_on="CPA Pretty Well ID")
    # using indicator to store which rows to append so that there are no overlaps with wells that are already in data

    oil_ab_append_df = oil_ab_append_df. \
        rename(columns={"CPA Pretty Well ID": "Well ID", "Reference (KB) Elev. (m)": "Reference Elev (m)",
                        "Surf-Hole Latitude (NAD83)": "Assigned Lattitude", "Test Date_y": "Test Date",
                        "Surf-Hole Longitude (NAD83)": "Assigned Longitude", "Pool Name": "Formation Tested",
                        })  # note Test Date_y from overlapping names in merge
    oil_ab_append_df = oil_ab_append_df[oil_ab_append_df["_merge"] == "right_only"]  # only rows that should be appended

    oil_ab_append_df = oil_ab_append_df.dropna(axis="columns", how="all")  # removing extra columns from the merge
    # the dropna is required so that duplicate column names are removed
    # it is a lazy method and might break the program if moved around

    oil_ab_append_df = oil_ab_append_df.fillna({
        "Datum Depth Pressure kPa":
            (abs(oil_ab_append_df["Pool Datum m"]) + abs(oil_ab_append_df["Kelly Bushing m"]) -
             abs(oil_ab_append_df["Run Depth m"])) * oil_ab_append_df["Gas Gradient"] + oil_ab_append_df[
                "Run Depth Pressure kPa"]})
    # filling empty datum depth pressure values using the provided formula (when the values in the formula are given)

    oil_ab_append_df["Pressure Measurement Elevation (m)"] = abs(oil_ab_append_df["Pool Datum m"]) \
        + abs(oil_ab_append_df["Reference Elev (m)"])
    # calculating pressure measurement elevation from the absolute value of datum depth plus reference elevation

    oil_ab_append_df["Pressure/Depth (kPa/m)"] = oil_ab_append_df["Datum Depth Pressure kPa"] / oil_ab_append_df[
        "Pressure Measurement Elevation (m)"]
    # calculating Pressure/Depth (kPa/m) from Datum Depth Pressure kPa and Pressure Measurement Elevation (m)

    oil_ab_append_df["Assigned Longitude"] = "-" + oil_ab_append_df["Assigned Longitude"].str.slice(stop=-1)
    oil_ab_append_df["Assigned Lattitude"] = oil_ab_append_df["Assigned Lattitude"].str.slice(stop=-1)
    # changing formatting of longitude and latitude so that they are displayed consistently (eg. 119.72W -> -119.72)

    oil_ab_append_df = oil_ab_append_df.rename(columns={"Pool Datum m": "Pool Datum m SL"})  # matching gas AB

    oil_ab_append_df = oil_ab_append_df[["Well ID", "Reference Elev (m)", "Test Date", "Assigned Lattitude",
                                         "Assigned Longitude", "Formation Tested", "Pressure/Depth (kPa/m)",
                                         "Pressure Measurement Elevation (m)",
                                         "Pool Datum m SL", "Run Depth Pressure kPa", "Datum Depth Pressure kPa"]]
    # specified columns
    return oil_ab_append_df


def get_pressure_test_gas_ab(excluded_df, well_list_df):
    """Finds wells in Pressure test GAS AB.csv that have records in a well list,
         but only if they do not already have records in a given dataset.
         excluded_df is the Pandas dataframe that contains wells which should not be added (All_Pp_Grad_Data.csv).
         well_list_df is the Pandas dataframe that contains the well list (Well_List.xlsx)."""
    pressure_test_gas_ab_df = pd.read_csv("Pressure test GAS AB.csv")
    merged_gas_ab_well_list_df = pd.merge(well_list_df, pressure_test_gas_ab_df, how="inner",
                                          left_on="CPA Pretty Well ID", right_on="Unique Well ID")
    # wells from AB gas pressure tests that are in the well list

    pool_set = {"HALFWAY Y & DOIG J", "TD UND"}
    pool_prefixes = ("DOIG", "MONT")
    merged_gas_ab_well_list_df = merged_gas_ab_well_list_df.loc[
        merged_gas_ab_well_list_df["Pool Name"].isin(pool_set) |  # | instead of or needed
        merged_gas_ab_well_list_df["Pool Name"].str.startswith(pool_prefixes) |
        merged_gas_ab_well_list_df["Pool Name"].isnull()]
    # selecting wells that are in one of the given pool names or are in a pool that begins with a given prefix
    # or do not have a value (redundant in practice, all get eliminated later)

    gas_ab_append_df = pd.merge(excluded_df, merged_gas_ab_well_list_df, how="outer", indicator=True,
                                left_on="Well ID", right_on="CPA Pretty Well ID")
    # using indicator to store which rows to append so that there are no overlaps with wells that are already in data

    gas_ab_append_df = gas_ab_append_df. \
        rename(columns={"CPA Pretty Well ID": "Well ID", "Reference (KB) Elev. (m)": "Reference Elev (m)",
                        "Surf-Hole Latitude (NAD83)": "Assigned Lattitude", "Test Date_y": "Test Date",
                        "Surf-Hole Longitude (NAD83)": "Assigned Longitude", "Pool Name": "Formation Tested",
                        "Test Type Description": "Test Class"})  # note Test Date_y from overlapping names in merge
    gas_ab_append_df = gas_ab_append_df[gas_ab_append_df["_merge"] == "right_only"]  # only rows that should be appended
    gas_ab_append_df = gas_ab_append_df.dropna(axis="columns", how="all")  # removing extra columns from the merge
    # the dropna is required so that duplicate column names are removed
    # it is a lazy method and might break the program if moved around

    cols = ["Kelly Bushing m SL", "Pool Datum m SL", "Run Depth m GE", "Run Depth Pressure kPa",
            "Datum Depth Pressure kPa", "Gas Gradient kPa/m"]
    gas_ab_append_df[cols] = gas_ab_append_df[cols].replace(["0", 0], np.nan)
    # the alberta gas dataset uses zeroes as placeholders for missing values, replacing with blanks for standardisation

    gas_ab_append_df = gas_ab_append_df.fillna({
        "Datum Depth Pressure kPa":
            (abs(gas_ab_append_df["Pool Datum m SL"]) + abs(gas_ab_append_df["Kelly Bushing m SL"]) -
             abs(gas_ab_append_df["Run Depth m GE"])) * gas_ab_append_df["Gas Gradient kPa/m"] + gas_ab_append_df[
                "Run Depth Pressure kPa"]})
    # filling empty datum depth pressure values using the provided formula (when the values in the formula are given)

    gas_ab_append_df["Pressure Measurement Elevation (m)"] = abs(gas_ab_append_df["Pool Datum m SL"]) \
        + abs(gas_ab_append_df["Reference Elev (m)"])
    # TODO: look at this
    # calculating pressure measurement elevation from the absolute value of datum depth plus reference elevation

    gas_ab_append_df["Pressure/Depth (kPa/m)"] = gas_ab_append_df["Datum Depth Pressure kPa"] / gas_ab_append_df[
        "Pressure Measurement Elevation (m)"]
    # calculating Pressure/Depth (kPa/m) from Datum Depth Pressure kPa and Pressure Measurement Elevation (m)

    gas_ab_append_df["Assigned Longitude"] = "-" + gas_ab_append_df["Assigned Longitude"].str.slice(stop=-1)
    gas_ab_append_df["Assigned Lattitude"] = gas_ab_append_df["Assigned Lattitude"].str.slice(stop=-1)
    # changing formatting of longitude and latitude so that they are displayed consistently (eg. 119.72W -> -119.72)

    gas_ab_append_df = gas_ab_append_df[["Well ID", "Reference Elev (m)", "Test Date", "Assigned Lattitude",
                                         "Assigned Longitude", "Formation Tested", "Pressure/Depth (kPa/m)",
                                         "Pressure Measurement Elevation (m)", "Test Class",
                                         "Pool Datum m SL", "Run Depth Pressure kPa", "Datum Depth Pressure kPa"]]
    # specified columns
    return gas_ab_append_df


def get_pressure_test_gas_bc(excluded_df, well_list_df):
    """Finds wells in Pressure test GAS BC.csv that have records in a well list,
             but only if they do not already have records in a given dataset.
             excluded_df is the Pandas dataframe that contains wells which should not be added (All_Pp_Grad_Data.csv).
             well_list_df is the Pandas dataframe that contains the well list (Well_List.xlsx)."""
    pressure_test_gas_bc_df = pd.read_csv("Pressure test GAS BC.csv")
    merged_gas_bc_well_list_df = pd.merge(well_list_df, pressure_test_gas_bc_df, how="inner",
                                          left_on="CPA Pretty Well ID", right_on="Unique Well ID")
    # wells from BC gas pressure tests that are in the well list

    formation_td_set = {"TRmntny_L", "TRmontney", "TRdoig", "TRdoig_L", "PRbelloy"}
    prod_inject_formation_set = {"TRmontney", "TRdoig", "TRdoig_L", "PRbelloy", "TRdoig;TRmontney", "TRhalfway;TRdoig",
                                 "TRhalfway;TRmontney", "TRmntny_L", "TRmntny_M", "TRmntny_U"}
    merged_gas_bc_well_list_df = merged_gas_bc_well_list_df.loc[
        merged_gas_bc_well_list_df["Prod./Inject. Frmtn"].isin(prod_inject_formation_set) |  # | instead of or needed
        (merged_gas_bc_well_list_df["Formation@TD"].isin(formation_td_set) &  # & instead of and needed
         merged_gas_bc_well_list_df["Prod./Inject. Frmtn"].isnull())]
    # selecting wells that have one of the production/injection formations required, or wells that do not have any
    # production/injection formation and have one of the formation@td required

    merged_gas_bc_well_list_df = merged_gas_bc_well_list_df.fillna({
        "Prod./Inject. Frmtn": merged_gas_bc_well_list_df["Formation@TD"]})
    # merging formation columns by filling empty production/injection fields with formation@td values

    gas_bc_append_df = pd.merge(excluded_df, merged_gas_bc_well_list_df, how="outer", indicator=True,
                                left_on="Well ID", right_on="CPA Pretty Well ID")
    # using indicator to store which rows to append so that there are no overlaps with wells that are already in data

    gas_bc_append_df = gas_bc_append_df. \
        rename(columns={"CPA Pretty Well ID": "Well ID", "Reference (KB) Elev. (m)": "Reference Elev (m)",
                        "Pressure Survey Date": "Test Date", "Surf-Hole Latitude (NAD83)": "Assigned Lattitude",
                        "Surf-Hole Longitude (NAD83)": "Assigned Longitude", "Prod./Inject. Frmtn": "Formation Tested",
                        "Pressure Survey Type Code": "Test Class"})
    gas_bc_append_df = gas_bc_append_df[gas_bc_append_df["_merge"] == "right_only"]  # only rows that should be appended
    gas_bc_append_df = gas_bc_append_df.dropna(axis="columns", how="all")  # removing extra columns from the merge
    # the dropna is required so that duplicate column names are removed
    # it is a lazy method and will break the program if moved around

    gas_bc_append_df["Pressure Measurement Elevation (m)"] = abs(gas_bc_append_df["Pressure Survey Datum Depth m SL"]) \
        + abs(gas_bc_append_df["Reference Elev (m)"])
    # TODO: look at this
    # calculating pressure measurement elevation from the absolute value of datum depth plus reference elevation

    gas_bc_append_df["Pressure/Depth (kPa/m)"] = gas_bc_append_df["Datum Pressure kPa"] / gas_bc_append_df[
        "Pressure Measurement Elevation (m)"]
    # calculating Pressure/Depth (kPa/m) from Datum Pressure kPa and Pressure Measurement Elevation (m)

    gas_bc_append_df["Assigned Longitude"] = "-" + gas_bc_append_df["Assigned Longitude"].str.slice(stop=-1)
    gas_bc_append_df["Assigned Lattitude"] = gas_bc_append_df["Assigned Lattitude"].str.slice(stop=-1)
    # changing formatting of longitude and latitude so that they are displayed consistently (eg. 119.72W -> -119.72)

    gas_bc_append_df = gas_bc_append_df.rename(columns={"Pressure Survey Datum Depth m SL": "Pool Datum m SL",
                                                        "Datum Pressure kPa": "Datum Depth Pressure kPa"})
    # matching gas AB
    # TODO: look at this

    gas_bc_append_df = gas_bc_append_df[["Well ID", "Reference Elev (m)", "Test Date", "Assigned Lattitude",
                                         "Assigned Longitude", "Formation Tested", "Pressure/Depth (kPa/m)",
                                         "Pressure Measurement Elevation (m)", "Test Class",
                                         "Pool Datum m SL", "Run Depth Pressure kPa", "Datum Depth Pressure kPa"]]
    # specified columns
    # TODO: look at this
    return gas_bc_append_df


def get_all_pp_grad_data(well_list_df):
    """Finds wells in All_Pp_Grad_Data.csv that have records in Well_List.xlsx"""
    all_pp_grad_data_df = pd.read_csv("All_Pp_Grad_Data.csv")
    cols = ["Pressure/Depth (kPa/m)"]
    all_pp_grad_data_df[cols] = all_pp_grad_data_df[cols].replace(["0", 0], np.nan)
    # the dataset uses zeroes as placeholders for missing pressure/depth, replacing with blanks for standardisation

    new_df = pd.merge(well_list_df, all_pp_grad_data_df, how="inner", left_on="CPA Pretty Well ID", right_on="Well ID")
    # inner join selects only wells that are in both of the datasets

    new_df = new_df[["Well ID", "Use in Montney Interval PD Map? (2=All, 1.5=Extra point, 1 = Redundant, 0 = No)",
                     "Test Class", "Reference Elev (m)", "Test Date", "Formation Tested",
                     "Pressure Measurement Elevation (m)", "Pressure/Depth (kPa/m)", "Test Remarks",
                     "Assigned Longitude", "Assigned Lattitude"]]  # selecting only the columns specified
    return new_df


def main():
    # noinspection PyArgumentList
    well_list_df = pd.read_excel("Well_List.xlsx", skiprows=[0, 2])  # skipping blank rows
    all_pp_grad_data_df = get_all_pp_grad_data(well_list_df)

    pressure_test_gas_bc_df = get_pressure_test_gas_bc(all_pp_grad_data_df, well_list_df)
    pressure_test_gas_ab_df = get_pressure_test_gas_ab(all_pp_grad_data_df, well_list_df)
    pressure_test_oil_ab_df = get_pressure_test_oil_ab(all_pp_grad_data_df, well_list_df)
    # all three extracted pressure tests from BC and AB only take wells that are in well_list but not all_pp_grad_data

    new_df = pd.concat([all_pp_grad_data_df, pressure_test_gas_bc_df, pressure_test_gas_ab_df, pressure_test_oil_ab_df],
                       join="outer")  # appending datasets together
    new_df = new_df.dropna(subset=["Pressure/Depth (kPa/m)"])
    # dropping entries without pressure/depth values

    new_df["Pressure Measurement Elevation (m)"] = pd.to_numeric(new_df["Pressure Measurement Elevation (m)"])
    new_df["Pool Datum m SL"] = pd.to_numeric(new_df["Pool Datum m SL"])
    new_df["Pressure/Depth (kPa/m)"] = pd.to_numeric(new_df["Pressure/Depth (kPa/m)"])
    # converting strings to integers or floats

    new_df = new_df.fillna({
        "Pool Datum m SL": new_df["Pressure Measurement Elevation (m)"] + new_df["Reference Elev (m)"],
        "Datum Depth Pressure kPa": abs(new_df["Pressure/Depth (kPa/m)"] * new_df["Pressure Measurement Elevation (m)"])
    })
    # TODO: look at this
    # calculating missing datum depth from pressure measurement elevation plus reference elevation
    # calculating missing datum depth pressure from (pressure/depth)*depth

    new_df["Pool Datum m SL"] = -abs(new_df["Pool Datum m SL"])  # TODO: look at this
    # standardising pool datum to be negative
    new_df["Pressure Measurement Elevation (m)"] = -abs(new_df["Pressure Measurement Elevation (m)"])
    # TODO: look at this
    # standardising pressure measurement elevation to be negative

    new_df.to_csv("Output.csv", encoding="utf-8", index=False)  # new data saved as Output.csv


if __name__ == "__main__":
    main()
