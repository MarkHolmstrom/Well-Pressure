#!/usr/bin/env python3

import os.path
import sys

import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog


def get_pressure_test_data(path, excluded_df, well_list_df):
    """Finds wells in a dataset that have records in a well list,
             but only if they do not already have records in a given dataset.
             excluded_df is the Pandas dataframe that contains wells which should not be added (All_Pp_Grad_Data.csv).
             well_list_df is the Pandas dataframe that contains the well list (Well_List.xlsx)."""

    if os.path.splitext(path)[1] in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls", ".xlt", ".xlm"}:
        # noinspection PyArgumentList
        pressure_test_df = pd.read_excel(path)
    else:  # currently only handling excel and comma separated files
        try:
            pressure_test_df = pd.read_csv(path)
        except UnicodeDecodeError:
            pressure_test_df = pd.read_csv(path, encoding="cp1252")
            # Pandas does not infer the correct encoding so try except blocks are necessary

    unique_well_id_col = input("Paste the column name that corresponds to Unique Well ID: ")
    pressure_test_df = pressure_test_df.rename(columns={unique_well_id_col: "Unique Well ID"})
    merged_pressure_test_well_list_df = pd.merge(well_list_df, pressure_test_df, how="inner",
                                                 left_on="CPA Pretty Well ID", right_on="Unique Well ID")
    # wells from pressure tests that are in the well list

    pool_name_col = input("Paste the column name that corresponds to Pool Name (enter blank if pool name is to be "
                          "gotten from the well list like Pressure test GAS BC.csv): ")
    if pool_name_col != "":
        merged_pressure_test_well_list_df = merged_pressure_test_well_list_df.rename(
            columns={pool_name_col: "Pool Name"})

        pools_exact = input("Enter the exact names of the pools that will be looked for in Pool Name "
                            "separated by commas: ")
        pools_exact = tuple([x.strip() for x in pools_exact.split(',')])
        pool_prefix = input("Enter the prefixes of the pools that will be looked for in Pool Name "
                            "separated by commas: ")
        pool_prefix = tuple([x.strip() for x in pool_prefix.split(',')])

        merged_pressure_test_well_list_df = merged_pressure_test_well_list_df.loc[
            merged_pressure_test_well_list_df["Pool Name"].isin(pools_exact) |  # | instead of or needed
            merged_pressure_test_well_list_df["Pool Name"].str.startswith(pool_prefix) |
            merged_pressure_test_well_list_df["Pool Name"].isnull()]
        # selecting wells that are in one of the given pool names or are in a pool that begins with a given prefix
        # or do not have a value (redundant in practice, all get eliminated later)
    else:
        formation_td_pools_exact = input("Enter the exact names of the pools that will be looked for in Formation@TD "
                                         "separated by commas: ")
        formation_td_pools_exact = tuple([x.strip() for x in formation_td_pools_exact.split(',')])
        formation_td_pools_prefix = input("Enter the prefixes of the pools that will be looked for in Formation@TD "
                                          "separated by commas: ")
        formation_td_pools_prefix = tuple([x.strip() for x in formation_td_pools_prefix.split(',')])

        prod_inject_exact = input("Enter the exact names of the pools that will be looked for in Prod./Inject. Frmtn "
                                  "separated by commas: ")
        prod_inject_exact = tuple([x.strip() for x in prod_inject_exact.split(',')])
        prod_inject_prefix = input("Enter the prefixes of the pools that will be looked for in Prod./Inject. Frmtn "
                                   "separated by commas: ")
        prod_inject_prefix = tuple([x.strip() for x in prod_inject_prefix.split(',')])

        merged_pressure_test_well_list_df = merged_pressure_test_well_list_df.loc[
            (merged_pressure_test_well_list_df["Prod./Inject. Frmtn"].isin(prod_inject_exact) |
             # | instead of or needed
             merged_pressure_test_well_list_df["Prod./Inject. Frmtn"].str.startswith(prod_inject_prefix)) |
            ((merged_pressure_test_well_list_df["Formation@TD"].isin(formation_td_pools_exact) |
              merged_pressure_test_well_list_df["Prod./Inject. Frmtn"].str.startswith(formation_td_pools_prefix)) &
             # & instead of and needed
             merged_pressure_test_well_list_df["Prod./Inject. Frmtn"].isnull())]
        # selecting wells that have one of the production/injection formations required, or wells that do not have any
        # production/injection formation and have one of the formation@td required

        merged_pressure_test_well_list_df = merged_pressure_test_well_list_df.fillna({
            "Prod./Inject. Frmtn": merged_pressure_test_well_list_df["Formation@TD"]})
        merged_pressure_test_well_list_df = merged_pressure_test_well_list_df.rename(
            columns={"Prod./Inject. Frmtn": "Pool Name"})
        # merging formation columns by filling empty production/injection fields with formation@td values

    append_df = pd.merge(excluded_df, merged_pressure_test_well_list_df, how="outer", indicator=True,
                         left_on="Well ID", right_on="CPA Pretty Well ID")
    # using indicator to store which rows to append so that there are no overlaps with wells that are already in data

    test_date_col = input("Paste the column name that corresponds to Pressure Survey Date: ")
    if test_date_col == "Test Date" and "Test Date_y" in append_df.columns:  # in case of common overlapping name
        append_df = append_df.rename(columns={"Test Date_y": "Test Date"})
    else:
        append_df = append_df.rename(columns={test_date_col: "Test Date"})

    append_df = append_df. \
        rename(columns={"CPA Pretty Well ID": "Well ID", "Reference (KB) Elev. (m)": "Reference Elev (m)",
                        "Surf-Hole Latitude (NAD83)": "Assigned Latitude", "Pool Name": "Formation Tested",
                        "Surf-Hole Longitude (NAD83)": "Assigned Longitude"})
    append_df = append_df[append_df["_merge"] == "right_only"]  # only rows that should be appended
    append_df = append_df.dropna(axis="columns", how="all")  # removing extra columns from the merge
    # the dropna is required so that duplicate column names are removed
    # it is a lazy method and might break the program if moved around, in particular columns intentionally having NaNs

    test_class_col = input("Paste the column name that corresponds to Test Class (enter blank if no such column): ")
    if test_class_col != "":
        append_df = append_df.rename(columns={test_class_col: "Test Class"})
    else:
        append_df["Test Class"] = np.nan  # this column would be removed by the dropna if the order was switched

    kb_elevation_col = input("Paste the column name that corresponds to Kelly Bushing m SL (enter blank if to be "
                             "gotten from well list): ")
    if kb_elevation_col != "":
        append_df = append_df.rename(columns={kb_elevation_col: "Kelly Bushing m SL"})
    else:
        append_df["Kelly Bushing m SL"] = append_df["Reference Elev (m)"]

    pool_datum_col = input("Paste the column name that corresponds to Pool Datum m SL: ")
    append_df = append_df.rename(columns={pool_datum_col: "Pool Datum m SL"})

    datum_depth_pressure_col = input("Paste the column name that corresponds to Datum Depth Pressure kPa: ")
    if "Datum Depth Pressure kPa_y" in append_df.columns:
        append_df = append_df.rename(columns={"Datum Depth Pressure kPa_y": "Datum Depth Pressure kPa"})
    else:
        append_df = append_df.rename(columns={datum_depth_pressure_col: "Datum Depth Pressure kPa"})
    # handling overlapping names in merge

    run_depth_col = input("Paste the column name that corresponds to Run Depth m GE (enter blank if no such column): ")
    if run_depth_col != "":
        append_df = append_df.rename(columns={run_depth_col: "Run Depth m GE"})
    else:
        append_df["Run Depth m GE"] = np.nan

    run_depth_pressure_col = input("Paste the column name that corresponds to Run Depth Pressure kPa (enter blank if "
                                   "no such column): ")
    if run_depth_pressure_col != "":
        append_df = append_df.rename(columns={run_depth_pressure_col: "Run Depth Pressure kPa"})
    else:
        append_df["Run Depth Pressure kPa"] = np.nan

    gas_gradient_col = input("Paste the column name that corresponds to Gas Gradient kPa/m (enter blank if no such "
                             "column): ")
    if gas_gradient_col != "":
        append_df = append_df.rename(columns={gas_gradient_col: "Gas Gradient kPa/m"})
    else:
        append_df["Gas Gradient kPa/m"] = np.nan

    cols = ["Kelly Bushing m SL", "Pool Datum m SL", "Run Depth m GE", "Run Depth Pressure kPa",
            "Datum Depth Pressure kPa", "Gas Gradient kPa/m"]
    append_df[cols] = append_df[cols].replace(["0", 0], np.nan)
    # some datasets use zeroes as placeholders for missing values, replacing with blanks for standardisation

    append_df = append_df.fillna({
        "Datum Depth Pressure kPa":
            (abs(append_df["Pool Datum m SL"]) + abs(append_df["Kelly Bushing m SL"]) -
             abs(append_df["Run Depth m GE"])) * append_df["Gas Gradient kPa/m"] + append_df["Run Depth Pressure kPa"]})
    # filling empty datum depth pressure values using the provided formula (when the values in the formula are given)

    append_df["Pool Datum m SL"] = -abs(append_df["Pool Datum m SL"])  # setting Pool Datum m SL to always be negative

    append_df["Datum (TVD) Depth (m)"] = append_df["Reference Elev (m)"] - append_df["Pool Datum m SL"]
    # calculating pressure measurement elevation from reference elevation minus the pool datum

    append_df["Pressure/Depth (kPa/m)"] = append_df["Datum Depth Pressure kPa"] / append_df["Datum (TVD) Depth (m)"]
    # calculating Pressure/Depth (kPa/m) from Datum Depth Pressure kPa and Pressure Measurement Elevation (m)

    append_df = append_df.rename(columns={"Pool Datum m SL": "Pressure Measurement Elevation (m)"})
    # pool datum is equivalent to pressure measurement elevation

    try:
        pd.to_numeric(append_df["Assigned Longitude"])
        pd.to_numeric(append_df["Assigned Latitude"])
    except ValueError:
        append_df["Assigned Longitude"] = "-" + append_df["Assigned Longitude"].str.slice(stop=-1)
        append_df["Assigned Latitude"] = append_df["Assigned Latitude"].str.slice(stop=-1)
        # changing formatting of longitude and latitude so that they are displayed consistently (eg. 119.72W -> -119.72)

    append_df = append_df[["Well ID", "Reference Elev (m)", "Test Date", "Assigned Latitude",
                           "Assigned Longitude", "Formation Tested", "Pressure/Depth (kPa/m)",
                           "Pressure Measurement Elevation (m)", "Test Class", "Run Depth Pressure kPa",
                           "Datum Depth Pressure kPa", "Datum (TVD) Depth (m)"]]
    # specified columns
    return append_df


def get_all_pp_grad_data(path, well_list_df):
    """Finds wells in a dataset similar to All_Pp_Grad_Data.csv that have records in a well list"""

    if path == "":
        sys.exit()

    if os.path.splitext(path)[1] in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls", ".xlt", ".xlm"}:
        # noinspection PyArgumentList
        all_pp_grad_data_df = pd.read_excel(path)
    else:  # currently only handling excel and comma separated files
        try:
            all_pp_grad_data_df = pd.read_csv(path)
        except UnicodeDecodeError:
            all_pp_grad_data_df = pd.read_csv(path, encoding="cp1252")
            # Pandas does not infer the correct encoding so try except blocks are necessary

    cols = ["Pressure/Depth (kPa/m)"]
    all_pp_grad_data_df[cols] = all_pp_grad_data_df[cols].replace(["0", 0], np.nan)
    # the dataset uses zeroes as placeholders for missing pressure/depth, replacing with blanks for standardisation

    well_id_col = input("Paste the column name that corresponds to Well ID: ")
    use_in_interval_col = input("Paste the column name that corresponds to Use in Montney Interval PD Map? (2=All, "
                                "1.5=Extra point, 1 = Redundant, 0 = No): ")
    test_class_col = input("Paste the column name that corresponds to Test Class: ")
    test_date_col = input("Paste the column name that corresponds to Test Date: ")
    formation_tested_col = input("Paste the column name that corresponds to Formation Tested: ")
    pressure_elevation_col = input("Paste the column name that corresponds to Pressure Measurement Elevation (m): ")
    pressure_depth_col = input("Paste the column name that corresponds to Pressure/Depth (kPa/m): ")
    test_remarks_col = input("Paste the column name that corresponds to Test Remarks: ")
    longitude_col = input("Paste the column name that corresponds to Assigned Longitude: ")
    latitude_col = input("Paste the column name that corresponds to Assigned Latitude: ")

    all_pp_grad_data_df = all_pp_grad_data_df. \
        rename(columns={well_id_col: "Well ID",
                        use_in_interval_col: "Use in Montney Interval PD Map? (2=All, 1.5=Extra point, 1 = Redundant, "
                                             "0 = No)",
                        test_class_col: "Test Class",
                        test_date_col: "Test Date",
                        formation_tested_col: "Formation Tested",
                        pressure_elevation_col: "Pressure Measurement Elevation (m)",
                        pressure_depth_col: "Pressure/Depth (kPa/m)",
                        test_remarks_col: "Test Remarks",
                        longitude_col: "Assigned Longitude",
                        latitude_col: "Assigned Latitude",
                        })
    new_df = pd.merge(well_list_df, all_pp_grad_data_df, how="inner", left_on="CPA Pretty Well ID", right_on="Well ID")
    # inner join selects only wells that are in both of the datasets

    new_df["Pressure Measurement Elevation (m)"] = pd.to_numeric(new_df["Pressure Measurement Elevation (m)"])
    new_df["Pressure/Depth (kPa/m)"] = pd.to_numeric(new_df["Pressure/Depth (kPa/m)"])
    new_df["Reference Elev (m)"] = pd.to_numeric(new_df["Reference Elev (m)"])
    # converting to numeric forms if Pandas interprets columns as strings

    new_df["Datum (TVD) Depth (m)"] = new_df["Reference Elev (m)"] - new_df["Pressure Measurement Elevation (m)"]
    new_df["Datum Depth Pressure kPa"] = new_df["Pressure/Depth (kPa/m)"] * new_df["Datum (TVD) Depth (m)"]

    new_df = new_df[["Well ID", "Use in Montney Interval PD Map? (2=All, 1.5=Extra point, 1 = Redundant, 0 = No)",
                     "Test Class", "Reference Elev (m)", "Test Date", "Formation Tested",
                     "Pressure Measurement Elevation (m)", "Datum (TVD) Depth (m)", "Pressure/Depth (kPa/m)",
                     "Datum Depth Pressure kPa", "Test Remarks", "Assigned Longitude", "Assigned Latitude"]]
    # selecting only the columns specified
    return new_df


def get_well_list(path):
    """Gets a well list from a file path"""

    if path == "":
        sys.exit()

    if os.path.splitext(path)[1] in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls", ".xlt", ".xlm"}:
        # noinspection PyArgumentList
        well_list_df = pd.read_excel(path)
    else:  # currently only handling excel and comma separated files
        try:
            well_list_df = pd.read_csv(path)
        except UnicodeDecodeError:
            well_list_df = pd.read_csv(path, encoding="cp1252")
            # Pandas does not infer the correct encoding so try except blocks are necessary

    well_list_df = well_list_df.replace(["0", 0], np.nan)  # replacing placeholder zeroes with NaN

    pretty_well_id_col = input("Paste the column name that corresponds to CPA Pretty Well ID: ")
    kb_elevation_col = input("Paste the column name that corresponds to Reference (KB) Elev. (m): ")
    formation_td_col = input("Paste the column name that corresponds to Formation@TD: ")
    prod_inject_frmtn_col = input("Paste the column name that corresponds to Prod./Inject. Frmtn: ")
    latitude_col = input("Paste the column name that corresponds to Surf-Hole Latitude (NAD83): ")
    longitude_col = input("Paste the column name that corresponds to Surf-Hole Longitude (NAD83): ")

    well_list_df = well_list_df. \
        rename(columns={pretty_well_id_col: "CPA Pretty Well ID", kb_elevation_col: "Reference (KB) Elev. (m)",
                        formation_td_col: "Formation@TD", prod_inject_frmtn_col: "Prod./Inject. Frmtn",
                        latitude_col: "Surf-Hole Latitude (NAD83)", longitude_col: "Surf-Hole Longitude (NAD83)",
                        })
    well_list_df = well_list_df[["CPA Pretty Well ID", "Reference (KB) Elev. (m)", "Formation@TD",
                                 "Prod./Inject. Frmtn", "Surf-Hole Latitude (NAD83)", "Surf-Hole Longitude (NAD83)"]]
    return well_list_df


def main():
    app = QApplication(sys.argv)  # just to keep QApplication in memory, a gui event loop with exec_() isn't needed

    print("Select the file that contains the well list, the first line should have the column names and there should "
          "not be any other rows that are not part of the data")
    well_list_path = QFileDialog().getOpenFileName(filter="All Files (*)")[0]
    well_list_df = get_well_list(well_list_path)

    print("Select the file that contains pressure tests on well IDs which will be exclusive to this file (like "
          "All_Pp_Grad_Data.csv), the first line should have the column names and there should not be any other rows "
          "that are not part of the data")
    all_pp_grad_data_path = QFileDialog().getOpenFileName(filter="All Files (*)")[0]
    all_pp_grad_data_df = get_all_pp_grad_data(all_pp_grad_data_path, well_list_df)

    pressure_test_list = [all_pp_grad_data_df]

    pressure_test_path = None
    while pressure_test_path != "":
        print("Select a file that contain pressure tests on well IDs which could be in other files (like "
              "Pressure test GAS BC.csv), the first line should have the column names and there should not be any "
              "other rows that are not part of the data. Select cancel to stop adding new files and save the output")
        pressure_test_path = QFileDialog().getOpenFileName(filter="All Files (*)")[0]
        if pressure_test_path == "":
            break  # handling exit here instead of in get_pressure_test_data() for convenience
        pressure_test_df = get_pressure_test_data(pressure_test_path, all_pp_grad_data_df, well_list_df)
        pressure_test_list.append(pressure_test_df)

    new_df = pd.concat(pressure_test_list, join="outer")  # appending datasets together
    new_df = new_df.dropna(subset=["Pressure/Depth (kPa/m)"])
    # dropping entries without pressure/depth values

    new_df["Pressure Measurement Elevation (m)"] = pd.to_numeric(new_df["Pressure Measurement Elevation (m)"])
    new_df["Pressure/Depth (kPa/m)"] = pd.to_numeric(new_df["Pressure/Depth (kPa/m)"])
    # converting strings to integers or floats

    new_df = new_df.fillna({
        "Datum Depth Pressure kPa": new_df["Pressure/Depth (kPa/m)"] * new_df["Datum (TVD) Depth (m)"]
    })  # filling in messing Datum Depth Pressure kPa values as necessary

    if "Run Depth Pressure kPa" not in new_df:
        new_df = new_df[["Well ID", "Use in Montney Interval PD Map? (2=All, 1.5=Extra point, 1 = Redundant, 0 = No)",
                         "Test Class", "Reference Elev (m)", "Test Date", "Formation Tested",
                         "Pressure Measurement Elevation (m)", "Datum (TVD) Depth (m)", "Pressure/Depth (kPa/m)",
                         "Datum Depth Pressure kPa", "Test Remarks", "Assigned Latitude", "Assigned Longitude"]]
        # if the only files processed are well_list and all_pp then this column does not exist
    else:
        new_df = new_df[["Well ID", "Use in Montney Interval PD Map? (2=All, 1.5=Extra point, 1 = Redundant, 0 = No)",
                         "Test Class", "Reference Elev (m)", "Test Date", "Formation Tested",
                         "Pressure Measurement Elevation (m)", "Datum (TVD) Depth (m)", "Pressure/Depth (kPa/m)",
                         "Datum Depth Pressure kPa", "Run Depth Pressure kPa", "Test Remarks", "Assigned Latitude",
                         "Assigned Longitude"]]  # reordering

    print("Select the file which to save the resulting data as")
    file_name_output = QFileDialog.getSaveFileName(filter="CSV Files (*.csv);;All Files (*)")[0]
    if file_name_output == "":
        sys.exit()
    new_df.to_csv(file_name_output, encoding="utf-8", index=False)


if __name__ == "__main__":
    main()
