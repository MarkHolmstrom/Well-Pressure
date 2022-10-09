#!/usr/bin/env python3

import pandas as pd


def task2(docs_with_dfit_df, well_list_df, appendix_a1_names, dfit_wells_checked_names):
    output2_df = docs_with_dfit_df[docs_with_dfit_df["CPA Well ID"].isin(well_list_df["CPA Well ID"])]

    output2_df = output2_df.merge(well_list_df, on="CPA Well ID")

    output2_df = output2_df[~output2_df["CPA Pretty Well ID"].isin(appendix_a1_names).values
                            & ~output2_df["CPA Pretty Well ID"].isin(dfit_wells_checked_names).values]  # and not in

    output2_df = output2_df[["CPA Pretty Well ID", "WA", "Area_x", "File name",
                             "Bot-Hole Latitude (NAD83)", "Bot-Hole Longitude (NAD83)"]]

    output2_df = output2_df.rename(columns={"CPA Pretty Well ID": "UWI", "Area_x": "Area"})

    output2_df.to_csv("Output_2.csv", encoding="utf-8", index=False)  # new data saved as Output_2.csv


def task1(appendix_a1_df, well_list_names, dfit_wells_checked_names):
    output1_df = appendix_a1_df[appendix_a1_df["UWI"].isin(well_list_names.values)
                                & ~appendix_a1_df["UWI"].isin(dfit_wells_checked_names.values)]  # and not in

    output1_df = output1_df[["UWI", "Midpoint Test Interval - Assigned (m  - TVD)", "Quality Code", "Test Date",
                             "Formation Tested (Abra - EGL)", "P* (linear) - Assigned (kPa)",
                             "Closure - Assigned (kPa)",
                             "Closure Gradient - Assigned (kPa/m)", "BH Longitude", "BH Latitude"]]

    output1_df = output1_df.rename(columns={"Midpoint Test Interval - Assigned (m  - TVD)":
                                            "Midpoint Test Interval - Assigned (m - TVD)"})  # weird character encoding

    output1_df.to_csv("Output_1.csv", encoding="utf-8", index=False)  # new data saved as Output_1.csv


def main():
    appendix_a1_df = pd.read_excel("Appendix A1 - DFIT Database v 05-13-2020 with redactions.xlsx", header=3)
    try:
        well_list_df = pd.read_csv("Well_List.csv")
    except UnicodeDecodeError:
        well_list_df = pd.read_csv("Well_List.csv", encoding="cp1252")
    dfit_wells_checked_df = pd.read_excel("DFIT_wells_checked_small_area.xlsx", header=1)
    docs_with_dfit_df = pd.read_excel("Docs with DFIT or MiniFrac in Name.xlsx")

    well_list_names = well_list_df["CPA Pretty Well ID"]
    dfit_wells_checked_names = dfit_wells_checked_df["UWI"]

    task1(appendix_a1_df, well_list_names, dfit_wells_checked_names)

    appendix_a1_names = appendix_a1_df["UWI"]

    task2(docs_with_dfit_df, well_list_df, appendix_a1_names, dfit_wells_checked_names)


if __name__ == "__main__":
    main()
