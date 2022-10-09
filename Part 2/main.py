#!/usr/bin/env python3

import os.path
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, TextBox
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog


def save_pore_pressure(pp_soliman, pp_nolte, pp_soliman_radial, pp_nolte_radial):
    """Saves pore pressure values found by the program. The expected inputs are from the plot_pore_pressure function
        and the output is saved as a CSV file through a system dialog."""
    print("Select the file which to save the resulting data as")
    file_name_output = QFileDialog.getSaveFileName(filter="CSV Files (*.csv);;All Files (*)")[0]
    if file_name_output == "":
        return
    new_df = pd.DataFrame([pp_soliman, pp_nolte, pp_soliman_radial, pp_nolte_radial], columns=["Pore Pressure (kPa)"])
    new_df["Label"] = ["Soliman", "Nolte", "Soliman Radial", "Nolte Radial"]  # adding corresponding labels
    new_df.to_csv(file_name_output, encoding="utf-8", index=False)


def plot_pore_pressure(df, start_point, isip_point, closure_point):
    """Displays the line plots of pressure over time. Expected input df is a Pandas Dataframe that has cumulative time
        as its first column and pressure as its second column.
        The expected inputs start_point, isip_point, closure_point are from their respective functions.
        The function returns pore pressure as manually adjusted by the user in the plots by clicking anywhere on them
        to select the pore pressure."""

    class MyPlot:  # using classes for object referencing convenience

        def __init__(self):
            self.fig, ax_dictionary = plt.subplot_mosaic([["1", "1"], ["2", "3"], ["4", "5"]])
            self.ax1 = ax_dictionary["1"]
            self.ax2 = ax_dictionary["2"]
            self.ax3 = ax_dictionary["3"]
            self.ax4 = ax_dictionary["4"]
            self.ax5 = ax_dictionary["5"]  # unpacking axes from dictionary

            self.press = False
            self.move = False
            self.fig.canvas.mpl_connect("button_press_event", self.onpress)
            self.fig.canvas.mpl_connect("button_release_event", self.onrelease)
            self.fig.canvas.mpl_connect("motion_notify_event", self.onmove)
            # creative code that avoids directly checking for zoom selection from https://stackoverflow.com/a/48452190
            # by ImportanceOfBeingErnest
            # CC BY-SA 3.0

            self.df = df
            self.start_point = start_point
            self.isip_point = isip_point
            self.closure_point = closure_point

            self.tp_injection_time = self.isip_point[0] - self.start_point[0]
            self.tc_closure_start = self.closure_point[0] - self.start_point[0]
            # polynomial fitting can't handle non-numbers, also denotes closure time
            # be very careful with indexing, have to subtract start_point.name index to get closure pressure index
            # when referencing series created in this function

            self.calc_stats()

            self.setup_plot1()
            self.setup_plot2()
            self.setup_plot3()
            self.setup_plot4()
            self.setup_plot5()

        def calc_stats(self):
            with np.errstate(invalid="ignore"):  # avoid printing RunTimeWarning on invalid mathematical operations
                self.delta_t = self.df.iloc[self.start_point.name:, 0] - self.isip_point[0]
                self.tt = self.df.iloc[self.start_point.name:, 0] - self.start_point[0]
                self.f_soliman = np.sqrt(1 / (self.tp_injection_time + self.delta_t))
                self.f_nolte = np.sqrt(
                    1 + np.pi ** 2 * (self.tt - self.tc_closure_start) / 16 / self.tc_closure_start) - \
                               np.sqrt(np.pi ** 2 * (self.tt - self.tc_closure_start) / 16 / self.tc_closure_start)
                self.f_s_radial = 1 / (self.delta_t + self.tp_injection_time)
                self.f_n_radial = 1 + 16 / np.pi ** 2 * self.tc_closure_start / (self.tt - self.tc_closure_start)

            self.nan_count = self.f_nolte[self.f_nolte == 1].index[0]
            self.nan_count_original = self.nan_count

        def setup_plot1(self):
            self.ax1.clear()
            self.ax1.set_xlabel(self.df.columns[0])
            self.ax1.set_ylabel(self.df.columns[1])
            self.ax1.plot(self.df.iloc[self.start_point.name:, 0], self.df.iloc[self.start_point.name:, 1], marker=".")
            self.ax1.callbacks.connect("xlim_changed", self.on_xlims_change)

        def setup_plot2(self, right_lim=0):
            end = len(self.f_soliman) - right_lim
            self.ax2.clear()
            poly_soliman = np.polynomial.polynomial.Polynomial.fit(
                self.f_soliman[self.nan_count - self.start_point.name:end],
                self.df.iloc[self.nan_count:len(self.df) - right_lim, 1],
                deg=1).convert().coef
            # somewhat convoluted approach with a nan counter and right limit parameter used to avoid recalculating
            # stats of differently-sized series
            # a possibly more readable solution would be to add the statistics as columns to the original dataframe
            self.pp_soliman = poly_soliman[0]  # polynomial constant coefficient
            # linear regression on Soliman equivalent time

            self.ax2.set_xlabel("Equivalent time, F (Soliman)")
            self.ax2.set_ylabel(self.df.columns[1])
            self.ax2.plot(self.f_soliman[self.nan_count - self.start_point.name:end],
                          self.df.iloc[self.nan_count:len(self.df) - right_lim, 1],
                          marker=".")
            self.ax2.set_xlim((self.ax2.get_xlim()[::-1][0], min(0, self.ax2.get_xlim()[::-1][1])))
            # invert x-axis and set right limit to be 0 if y-intercept is not in the default limit

            self.ax2.plot([0, self.f_soliman[self.nan_count]],
                          [self.pp_soliman, self.pp_soliman + poly_soliman[1] * self.f_soliman[self.nan_count]])

            self.ax2_closure_marker = self.ax2.axhline(self.pp_soliman, color="tab:green", linestyle="--")
            self.ax2_closure_marker2 = self.ax1.axhline(self.pp_soliman, color="tab:green", linestyle="--")
            self.ax2_zero_intercept = self.ax2.axvline(0, color="grey", linewidth=1)
            self.ax2.callbacks.connect("xlim_changed", self.on_xlims_change)

        def setup_plot3(self, right_lim=0):
            end = len(self.f_nolte) - right_lim
            self.ax3.clear()
            poly_nolte = np.polynomial.polynomial.Polynomial.fit(
                self.f_nolte[self.nan_count - self.start_point.name:end],
                self.df.iloc[self.nan_count:len(self.df) - right_lim, 1],
                deg=1).convert().coef
            self.pp_nolte = poly_nolte[0]  # polynomial constant coefficient
            # linear regression on Nolte equivalent time
            self.ax3.set_xlabel("Equivalent time, F (Nolte)")
            self.ax3.set_ylabel(self.df.columns[1])
            self.ax3.plot(self.f_nolte[self.nan_count - self.start_point.name:end],
                          self.df.iloc[self.nan_count:len(self.df) - right_lim, 1],
                          marker=".")
            self.ax3.set_xlim((self.ax3.get_xlim()[::-1][0], min(0, self.ax3.get_xlim()[::-1][1])))
            # invert x-axis and set right limit to be 0 if y-intercept is not in the default limit

            self.ax3.plot([0, self.f_nolte[self.nan_count]],
                          [self.pp_nolte, self.pp_nolte + poly_nolte[1] * self.f_nolte[self.nan_count]])

            self.ax3_closure_marker = self.ax3.axhline(self.pp_nolte, color="tab:red", linestyle="--")
            self.ax3_closure_marker2 = self.ax1.axhline(self.pp_nolte, color="tab:red", linestyle="--")
            self.ax3_zero_intercept = self.ax3.axvline(0, color="grey", linewidth=1)
            self.ax3.callbacks.connect("xlim_changed", self.on_xlims_change)

        def setup_plot4(self, right_lim=0):
            end = len(self.f_s_radial) - right_lim
            self.ax4.clear()
            poly_soliman_r = np.polynomial.polynomial.Polynomial.fit(
                self.f_s_radial[self.nan_count - self.start_point.name:end],
                self.df.iloc[self.nan_count:len(self.df) - right_lim, 1],
                deg=1).convert().coef
            self.pp_soliman_r = poly_soliman_r[0]  # polynomial constant coefficient
            # linear regression on Soliman (radial) equivalent time

            self.ax4.set_xlabel("Equivalent time, F (Soliman Radial)")
            self.ax4.set_ylabel(self.df.columns[1])
            self.ax4.plot(self.f_s_radial[self.nan_count - self.start_point.name:end],
                          self.df.iloc[self.nan_count:len(self.df) - right_lim, 1],
                          marker=".")

            self.ax4.set_xlim((self.ax4.get_xlim()[::-1][0], min(0, self.ax4.get_xlim()[::-1][1])))
            # invert x-axis and set right limit to be 0 if y-intercept is not in the default limit

            self.ax4.plot([0, self.f_s_radial[self.nan_count]],
                          [self.pp_soliman_r, self.pp_soliman_r + poly_soliman_r[1] * self.f_s_radial[self.nan_count]])

            self.ax4_closure_marker = self.ax4.axhline(self.pp_soliman_r, color="tab:purple", linestyle="--")
            self.ax4_closure_marker2 = self.ax1.axhline(self.pp_soliman_r, color="tab:purple", linestyle="--")
            self.ax4_zero_intercept = self.ax4.axvline(0, color="grey", linewidth=1)
            self.ax4.callbacks.connect("xlim_changed", self.on_xlims_change)

        def setup_plot5(self, right_lim=0):
            end = len(self.f_n_radial) - right_lim
            self.ax5.clear()
            poly_nolte_r = np.polynomial.polynomial.Polynomial.fit(
                np.log(self.f_n_radial[self.nan_count - self.start_point.name + 1:end]),
                self.df.iloc[self.nan_count + 1:len(self.df) - right_lim, 1],
                deg=1).convert().coef
            self.pp_nolte_r = poly_nolte_r[0]
            # linear regression on Nolte (radial) equivalent time (logged)
            # adding one to skip division by 0 value

            self.ax5.set_xlabel("Equivalent time, F (Nolte Radial)")
            self.ax5.set_ylabel(self.df.columns[1])
            self.ax5.plot(self.f_n_radial[self.nan_count - self.start_point.name + 1:end],
                          self.df.iloc[self.nan_count + 1:len(self.df) - right_lim, 1],
                          marker=".")
            self.ax5.set_xscale("log")
            self.ax5.set_xlim((self.ax5.get_xlim()[::-1][0], min(1, self.ax5.get_xlim()[::-1][1])))
            # invert x-axis and set right limit to be 1 if y-intercept is not in the default limit
            # adding one to skip division by 0 value

            self.ax5.plot([1, self.f_n_radial[self.nan_count + 1]],
                          [self.pp_nolte_r,
                           self.pp_nolte_r + poly_nolte_r[1] * np.log(self.f_n_radial[self.nan_count + 1])])
            # log the inner multiplication self.f_n_radial term instead of the whole expression

            self.ax5_closure_marker = self.ax5.axhline(self.pp_nolte_r, color="tab:brown", linestyle="--")
            self.ax5_closure_marker2 = self.ax1.axhline(self.pp_nolte_r, color="tab:brown", linestyle="--")
            self.ax5_zero_intercept = self.ax5.axvline(1, color="grey", linewidth=1)  # helpful vertical lines
            self.ax5.callbacks.connect("xlim_changed", self.on_xlims_change)

        def onclick(self, event):
            if event.inaxes == self.ax2:
                self.ax2_closure_marker.remove()
                self.ax2_closure_marker2.remove()
                self.pp_soliman = event.ydata
                self.ax2_closure_marker = self.ax2.axhline(self.pp_soliman, color="tab:green", linestyle="--")
                self.ax2_closure_marker2 = self.ax1.axhline(self.pp_soliman, color="tab:green", linestyle="--")
            elif event.inaxes == self.ax3:
                self.ax3_closure_marker.remove()
                self.ax3_closure_marker2.remove()
                self.pp_nolte = event.ydata
                self.ax3_closure_marker = self.ax3.axhline(self.pp_nolte, color="tab:red", linestyle="--")
                self.ax3_closure_marker2 = self.ax1.axhline(self.pp_nolte, color="tab:red", linestyle="--")
            elif event.inaxes == self.ax4:
                self.ax4_closure_marker.remove()
                self.ax4_closure_marker2.remove()
                self.pp_soliman_r = event.ydata
                self.ax4_closure_marker = self.ax4.axhline(self.pp_soliman_r, color="tab:purple", linestyle="--")
                self.ax4_closure_marker2 = self.ax1.axhline(self.pp_soliman_r, color="tab:purple", linestyle="--")
            elif event.inaxes == self.ax5:
                self.ax5_closure_marker.remove()
                self.ax5_closure_marker2.remove()
                self.pp_nolte_r = event.ydata
                self.ax5_closure_marker = self.ax5.axhline(self.pp_nolte_r, color="tab:brown", linestyle="--")
                self.ax5_closure_marker2 = self.ax1.axhline(self.pp_nolte_r, color="tab:brown", linestyle="--")
            else:
                return  # if no valid ax is clicked return instead of redrawing needlessly
            plt.draw()

        def onpress(self, event):
            self.press = True

        def onmove(self, event):
            if self.press:
                self.move = True

        def onrelease(self, event):
            if self.press and not self.move:
                self.onclick(event)
            self.press = False
            self.move = False

        def on_xlims_change(self, event):
            if self.press:  # only on user input to avoid infinite recursion loops
                self.press = False  # avoids infinite recursion loops when xlims are updated on plot setup
                if event == self.ax1:
                    self.nan_count = max(self.nan_count_original,
                                         len(self.df[self.df.iloc[:, 0].between(self.df.iloc[0, 0],
                                                                                max(event.get_xlim()[0],
                                                                                    self.df.iloc[
                                                                                        self.start_point.name, 0]))]))

                    right_lim = len(self.df[self.df.iloc[:, 0].between(event.get_xlim()[1], len(self.df.iloc[:, 0]))])
                    # right_lim <-- how many are excluded by right limit
                    # self.setup_plot1()  # no zoom when graph zooms
                    try:
                        self.ax2_closure_marker2.remove()
                        self.ax3_closure_marker2.remove()
                        self.ax4_closure_marker2.remove()
                        self.ax5_closure_marker2.remove()
                    except ValueError:  # if the closure markers don't exist, for whatever reason, ignore them
                        pass
                    # proper zoom with marker updates, comment out if calling setup_plot1 (not needed)
                    self.setup_plot2(right_lim)
                    self.setup_plot3(right_lim)
                    self.setup_plot4(right_lim)
                    self.setup_plot5(right_lim)
                elif event == self.ax2:
                    left_lim = max(len(self.f_soliman[event.get_xlim()[0] <= self.f_soliman]) + self.start_point.name,
                                   self.nan_count_original)
                    # left_lim <-- how many are excluded by left limit
                    # to not cause errors with trying to regress on nan data
                    right_lim = len(self.f_soliman[self.f_soliman <= event.get_xlim()[1]])
                    nan_placeholder = self.nan_count  # save the nan count and restore it later
                    self.nan_count = left_lim  # more convenient that writing a custom parameter to the setup function
                    try:
                        self.ax2_closure_marker2.remove()
                    except ValueError:  # if the closure marker doesn't exist, for whatever reason, ignore it
                        pass
                    self.setup_plot2(right_lim)
                    self.nan_count = nan_placeholder  # restore nan count to avoid unexpected behaviour later
                elif event == self.ax3:
                    left_lim = max(len(self.f_nolte[event.get_xlim()[0] <= self.f_nolte]) + self.nan_count_original,
                                   self.nan_count_original)
                    # left_lim <-- how many are excluded by left limitx
                    # to not cause errors with trying to regress on nan data
                    # HAVE TO USE NAN COUNT INSTEAD OF START POINT NAME UNLIKE SOLIMAN TIME
                    right_lim = len(self.f_nolte[self.f_nolte <= event.get_xlim()[1]])
                    nan_placeholder = self.nan_count  # save the nan count and restore it later
                    self.nan_count = left_lim  # more convenient that writing a custom parameter to the setup function
                    try:
                        self.ax3_closure_marker2.remove()
                    except ValueError:  # if the closure marker doesn't exist, for whatever reason, ignore it
                        pass
                    self.setup_plot3(right_lim)
                    self.nan_count = nan_placeholder  # restore nan count to avoid unexpected behaviour later
                elif event == self.ax4:
                    left_lim = max(len(self.f_s_radial[event.get_xlim()[0] <= self.f_s_radial]) + self.start_point.name,
                                   self.nan_count_original)
                    # left_lim <-- how many are excluded by left limit
                    # to not cause errors with trying to regress on nan data
                    right_lim = len(self.f_s_radial[self.f_s_radial <= event.get_xlim()[1]])
                    nan_placeholder = self.nan_count  # save the nan count and restore it later
                    self.nan_count = left_lim  # more convenient that writing a custom parameter to the setup function
                    try:
                        self.ax4_closure_marker2.remove()
                    except ValueError:  # if the closure marker doesn't exist, for whatever reason, ignore it
                        pass
                    self.setup_plot4(right_lim)
                    self.nan_count = nan_placeholder  # restore nan count to avoid unexpected behaviour later
                elif event == self.ax5:
                    left_lim = max(
                        len(self.f_n_radial[event.get_xlim()[0] <= self.f_n_radial]) + self.nan_count_original,
                        self.nan_count_original)
                    # left_lim <-- how many are excluded by left limit
                    # to not cause errors with trying to regress on nan data
                    # HAVE TO USE NAN COUNT INSTEAD OF START POINT NAME UNLIKE SOLIMAN TIME
                    right_lim = len(self.f_n_radial[self.f_n_radial <= event.get_xlim()[
                        1]]) - self.nan_count_original + self.start_point.name
                    # HAVE TO DO ADDITIONAL MATH WITH NAN COUNT AND START POINT UNLIKE ALL OTHERS
                    nan_placeholder = self.nan_count  # save the nan count and restore it later
                    self.nan_count = left_lim  # more convenient that writing a custom parameter to the setup function
                    try:
                        self.ax5_closure_marker2.remove()
                    except ValueError:  # if the closure marker doesn't exist, for whatever reason, ignore it
                        pass
                    self.setup_plot5(right_lim)
                    self.nan_count = nan_placeholder  # restore nan count to avoid unexpected behaviour later

        def show(self):
            plt.show()

    myplot = MyPlot()
    myplot.show()

    return myplot.pp_soliman, myplot.pp_nolte, myplot.pp_soliman_r, myplot.pp_nolte_r


def closure_point_outlier_removal(df, mode="G"):
    differential = "dP/d" + mode
    rolling_variance = df[differential].rolling(5, center=True).var()
    q1 = rolling_variance.quantile(0.25)
    q3 = rolling_variance.quantile(0.75)
    df_no_outliers = df[rolling_variance < q3 + 1.5 * (q3 - q1)]
    # using interquartile range of rolling variance to detect local outliers has the advantage of filtering values
    # caused by noise near outliers as well, alternatively:
    # df2 = df[df[differential].rolling(5, center=True).var() <
    #       df[differential].rolling(5, center=True).var().quantile(0.8)]
    closure_point = df.iloc[df_no_outliers[differential].idxmax()]
    # select closure point by finding maximum dP/dG or dP/dSQRT(t) value that isn't an outlier
    return closure_point


def get_closure_point(df, breakdown_point, isip_point, mode="G"):
    """Selects the closure point of DFIT injection. Expected input is a Pandas Dataframe that has cumulative time
        as its first column, pressure as its second column.
        The second expected input is the breakdown point, from the earlier function get_breakdown_point.
        The third expected input is the ISIP point, from the earlier function get_isip_point.
        The optional argument mode specifies whether the closure point should be selected from G-function calculations
        or square root time calculations.
        Adds (or modifies) six columns for closure point calculation ordered as in the Excel examples."""
    df_aug = df[isip_point.name:]  # only data after and including shut-in

    time_since_breakdown = df_aug.iloc[:, 0] - breakdown_point[0]
    dimensionless_time = (time_since_breakdown - time_since_breakdown.iloc[0]) / time_since_breakdown.iloc[0]
    delta_t_since_shut_in = time_since_breakdown - time_since_breakdown.iloc[0]
    df["SQRT(t)"] = np.sqrt(delta_t_since_shut_in)
    df["G"] = 4 / np.pi * (((1 + dimensionless_time) ** (3 / 2) - dimensionless_time ** (3 / 2)) * 4 / 3 - 4 / 3)
    df["SQRT(t)*dP/dSQRT(t)"] = (df["SQRT(t)"] * (df_aug.iloc[:, 1].diff()) / (df["SQRT(t)"].diff()) * -1)
    df["GdP/dG"] = df["G"] * (df_aug.iloc[:, 1].diff()) / (df["G"].diff()) * (-1)
    df["dP/dG"] = df["GdP/dG"] / df["G"]
    df["dP/dSQRT(t)"] = df["SQRT(t)*dP/dSQRT(t)"] / df["SQRT(t)"]
    # all formulas copied from Excel files

    closure_point = closure_point_outlier_removal(df, mode)
    return closure_point


def get_isip_point(df, breakdown_point, ref_pressure_ratio=9 / 10, ref_slope_ratio=1 / 4):
    """Selects the ISIP point of DFIT injection. Expected input is a Pandas Dataframe that has cumulative time
        as its first column, pressure as its second column, and slope as its third column.
        The second expected input is the breakdown point, from the earlier function get_breakdown_point.
        The optional argument ref_pressure_ratio (0-1) is used to determine the slope after breakdown,
        setting it too low can result in having a slope that measures from breakdown to a point after ISIP,
        while setting it too high can result in numerically unstable slopes.
        The optional argument ref_slope_ratio (0-1) is used to determine the deviation from the slope which marks the
        ISIP, the lower the reference slope ratio the further the ISIP is from the breakdown. If the curve is
        concave after breakdown the reference slope ratio should be greater than 1."""
    df_post = df[df.iloc[:, 0] > breakdown_point[0]]  # curve with only data after the breakdown

    slope_endpoint = df.iloc[(df_post.iloc[:, 1] < breakdown_point[1] * ref_pressure_ratio).idxmax()]
    # first measurement where pressure is less than that of the breakdown point times reference pressure ratio
    post_breakdown_slope = (slope_endpoint[1] - breakdown_point[1]) / (slope_endpoint[0] - breakdown_point[0])
    # pressure over time from breakdown to chosen point

    smoothed_slope = df.iloc[:, 1].rolling(5, center=True).mean().diff() / \
                     df.iloc[:, 0].rolling(5, center=True).mean().diff()
    # rolling window size of 5 with centering to smooth out data

    isip_point = df.iloc[((smoothed_slope > post_breakdown_slope * ref_slope_ratio) &
                          (df_post.iloc[:, 0] > slope_endpoint[0])).idxmax() - 1]
    # the first value after the endpoint where the smoothed slope is less than the post-breakdown slope times reference
    # slope ratio (shifted -1 to be on left endpoint)

    return isip_point


def get_breakdown_point(df):
    """Selects the breakdown point of DFIT injection. Expected input is a Pandas Dataframe that has
        pressure as its second column."""
    breakdown_point = df.iloc[df.iloc[:, 1].idxmax()]  # maximum pressure value
    return breakdown_point


def get_start_point(df, ref_slope=5):
    """Selects the start point of DFIT injection. Expected input is a Pandas Dataframe that has cumulative time
        as its first column, pressure as its second column, and slope as its third column.
        The optional argument ref_slope (0-inf) is used to determine the rate of change in pressure over time
        that marks the starting point, the higher the reference slope the faster the pressure needs to change before
        injection is considered started. Its units are kPa/s."""
    start_point = df.iloc[(df.iloc[:, 2] > ref_slope * 3600).idxmax() - 1]
    # first value where slope is greater than reference slope (shifted -1 to be on left endpoint)
    # corresponds to the point where the rate of change in pressure is greater than 5 kPa per second by default
    # this seems to be accurate enough on tested datasets, if data is too noisy then the value should be increased
    if (start_point == df.iloc[-1]).all():
        start_point = df.iloc[0]
        # reset start point to 0 if it is selected by argmax
        # this is ONLY possible if the reference slope threshold is never reached in the pressure test (zeroth is NaN)
        # the user would then expect to see the first point selected instead of the last one
    return start_point


def save_data_points(df, start_point, breakdown_point, isip_point, closure_point):
    """Saves time and pressure of data points found by the program. The expected inputs are from their respective
        functions and the output is saved as a CSV file through a system dialog.
        Point names are expended to be the indices of the Dataframe and first two columns of the Dataframe are expected
        to be time and pressure."""
    print("Select the file which to save the resulting data as "
          "or cancel to continue to pore pressure calculations without saving")
    file_name_output = QFileDialog.getSaveFileName(filter="CSV Files (*.csv);;All Files (*)")[0]
    if file_name_output == "":
        return
    new_df = df.iloc[[start_point.name, breakdown_point.name, isip_point.name, closure_point.name], [0, 1]]
    new_df["Label"] = ["Start", "Breakdown", "ISIP", "Closure"]  # adding corresponding labels
    new_df.to_csv(file_name_output, encoding="utf-8", index=False)


def plot_dfit(df, start_point, breakdown_point, isip_point, closure_point):
    """Displays the line plots of pressure over time. Expected input df is a Pandas Dataframe that has cumulative time
        as its first column and pressure as its second column.
        The expected inputs start_point, breakdown_point, isip_point, closure_point are from their respective functions.
        The function returns updated points as manually adjusted by the user in the plots by selecting the point to
        update with the radio button and clicking on a measurement to set the point to it, or in the case of the bottom
        plot by clicking anywhere on it to select the closure point."""

    class MyPlot:  # using classes for object referencing convenience

        def __init__(self):
            self.fig, (self.ax1, self.ax2) = plt.subplots(2)
            self.controlfig = plt.figure(figsize=[3.3, 2.4])
            self.ax3 = self.ax2.twinx()

            self.df = df
            self.start_point = start_point
            self.breakdown_point = breakdown_point
            self.isip_point = isip_point
            self.closure_point = closure_point

            # lambda used to reference parameterised function without calling it

            self.fig.canvas.mpl_connect("pick_event", self.onpick)  # for top plot

            self.press = False
            self.move = False
            self.fig.canvas.mpl_connect("button_press_event", self.onpress)
            self.fig.canvas.mpl_connect("button_release_event", self.onrelease)
            self.fig.canvas.mpl_connect("motion_notify_event", self.onmove)  # for bottom plot
            # creative code that avoids directly checking for zoom selection from https://stackoverflow.com/a/48452190
            # by ImportanceOfBeingErnest
            # CC BY-SA 3.0

            self.rax = self.controlfig.add_axes([0, 0.6, 1, 0.4])
            self.radio = RadioButtons(self.rax, ("1 - Start", "2 - Breakdown", "3 - ISIP", "4 - Closure", "None"))
            self.rax2 = self.controlfig.add_axes([0, 0, 1, 0.4])
            self.radio2 = RadioButtons(self.rax2, ("G", "SQRT(t)"))
            self.radio2.on_clicked(lambda x: self.radio2_clicked(mode=self.radio2.value_selected))

            self.tax = self.controlfig.add_axes([0, 0.5, 1, 0.1])
            self.text_pressure = TextBox(self.tax, "Pressure", textalignment="center",
                                         initial="Custom ISIP pressure (kPa): ")
            self.tax2 = self.controlfig.add_axes([0, 0.4, 1, 0.1])
            self.text_time = TextBox(self.tax2, "Time", textalignment="center",
                                     initial="Custom ISIP time (cumulative hours): ")

            self.text_pressure.on_submit(self.submit)
            self.text_time.on_submit(self.submit)

            self.used_df = self.df  # workaround to make interacting with ax2 to select closure work properly

            self.setup_plot1()
            self.setup_plot2(mode=self.radio2.value_selected)

        def radio2_clicked(self, mode):  # wrapper for setting up plot 2, (need to redraw it when button is clicked)
            self.setup_plot2(mode)
            plt.figure(self.fig.number)  # setting active figure to the one with data to draw it properly
            plt.draw()

        def submit(self, expression):  # not using expression parameter, getting values from both text boxes directly
            try:
                pressure = float([x.strip() for x in self.text_pressure.text.split(":")][-1])  # rightmost entry after :
            except ValueError:
                return  # return if invalid number (possibly not filled out)
            try:
                time = float([x.strip() for x in self.text_time.text.split(":")][-1])  # rightmost entry after :
            except ValueError:
                return  # return if invalid number (possibly not filled out)
            insert_index = self.df.iloc[(self.df.iloc[::-1, 0] <= time).idxmax()].name
            # index of last row which has a cumulative time value less than given one, i.e. rows after this one will
            # need to be reindexed
            if np.isclose((time, pressure), (self.df.iloc[insert_index, 0], self.df.iloc[insert_index, 1])).all():
                self.isip_point = self.df.iloc[insert_index]  # if an existing point is chosen, simply use it
            else:
                self.df.loc[insert_index + 0.5] = time, pressure, *tuple([np.nan for x in self.df.columns])[:-2]
                self.df = self.df.sort_index().reset_index(drop=True)
                # code to insert values with simple reindexing from https://stackoverflow.com/a/63736275
                # by SergioGM
                # CC BY-SA 4.0
                self.isip_point = self.df.iloc[insert_index + 1]
                for point in (self.start_point, self.breakdown_point, self.closure_point):
                    if point.name >= self.isip_point.name:
                        point.name += 1  # adjust other point indices if needed

            self.closure_point = get_closure_point(self.df, self.breakdown_point, self.isip_point,
                                                   mode=self.radio2.value_selected)
            self.used_df = self.df[self.isip_point.name:]  # for use in plot 2
            # recalculating closure point and associated stats automatically if custom ISIP point is selected
            self.setup_plot1()
            self.setup_plot2(mode=self.radio2.value_selected)
            plt.figure(self.fig.number)  # setting active figure to the one with data to draw it properly
            plt.draw()

        def setup_plot1(self):
            self.press = False  # avoid redrawing needlessly
            self.ax1.clear()
            self.ax1.set_xlabel(self.df.columns[0])
            self.ax1.set_ylabel(self.df.columns[1])
            self.line1, = self.ax1.plot(self.df.iloc[:, 0], self.df.iloc[:, 1], label="Pressure over time", marker=".",
                                        picker=True, pickradius=5)
            self.line1_1, = self.ax1.plot((self.breakdown_point[0], self.isip_point[0]),
                                          (self.breakdown_point[1], self.isip_point[1]), color="tab:red")
            self.start_annotation = self.ax1.annotate(1, (self.start_point[0], self.start_point[1]),
                                                      fontsize=15)
            self.breakdown_annotation = self.ax1.annotate(2, (self.breakdown_point[0], self.breakdown_point[1]),
                                                          fontsize=15)
            self.isip_annotation = self.ax1.annotate(3, (self.isip_point[0], self.isip_point[1]),
                                                     fontsize=15)
            self.closure_annotation = self.ax1.annotate(4, (self.closure_point[0], self.closure_point[1]),
                                                        fontsize=15)

            self.ax1.legend(loc="upper right")

            self.ax1.callbacks.connect("xlim_changed", self.on_xlims_change)

        def setup_plot2(self, mode):
            self.press = False  # avoid redrawing needlessly
            self.ax2.clear()
            self.ax3.clear()
            self.ax3.remove()
            # clear existing ax2 and ax3 because the plot may need to be redrawn entirely
            time_col, ax2_xlabel, ax3_ylabel, line3_col, line3_label, line4_col, line4_label = None, None, None, None, \
                                                                                               None, None, None
            if mode == "G":
                time_col = self.used_df.iloc[:, 4]
                ax2_xlabel = self.used_df.columns[4]
                ax3_ylabel = "GdP/dG, dP/dG"
                line3_col = self.used_df.iloc[:, 6]
                line3_label = "GdP/dG over time"
                line4_col = self.used_df.iloc[:, 7]
                line4_label = "dP/dG over time"
            elif mode == "SQRT(t)":
                time_col = self.used_df.iloc[:, 3]
                ax2_xlabel = self.used_df.columns[3]
                ax3_ylabel = "SQRT(t)*dP/dSQRT(t), dP/dSQRT(t)"
                line3_col = self.used_df.iloc[:, 5]
                line3_label = "SQRT(t)*dP/dSQRT(t) over time"
                line4_col = self.used_df.iloc[:, 8]
                line4_label = "dP/dSQRT(t) over time"

            self.ax2.set_xlabel(ax2_xlabel)
            self.ax2.set_ylabel(self.used_df.columns[1])
            self.ax3 = self.ax2.twinx()
            self.ax3.set_ylabel(ax3_ylabel)

            self.line3, = self.ax3.plot(time_col, line3_col, label=line3_label, marker=".",
                                        picker=True, pickradius=5, color="tab:orange")
            self.line4, = self.ax3.plot(time_col, line4_col, label=line4_label, marker=".",
                                        picker=True, pickradius=5, color="tab:gray")

            self.line2, = self.ax2.plot(time_col, self.used_df.iloc[:, 1], label="Pressure over time",
                                        marker=".", picker=True, pickradius=5)

            self.ax2_closure_marker = self.ax3.axvline(self.closure_point[mode], color="black", linestyle="--")

            self.ax2.set_ylim(0, None)
            # setting lower y-axis limit of Pressure vs G-time to 0 like Excel

            q1 = self.used_df.iloc[:, 6].quantile(0.25)
            q3 = self.used_df.iloc[:, 6].quantile(0.75)
            high = q3 + 3 * (q3 - q1)
            self.ax3.set_ylim(0, high)
            # setting upper y-axis limit of GdP/dG vs G-time via interquartile range, lower set to 0 like Excel
            # times 3 instead of 1.5 looks better

            lines, labels = self.ax2.get_legend_handles_labels()
            lines2, labels2 = self.ax3.get_legend_handles_labels()
            self.ax3.legend(lines + lines2, labels + labels2, loc="upper right")  # consolidate legend

            self.ax2.callbacks.connect("xlim_changed", self.on_xlims_change)

        def onpick(self, event):
            """Manually picking points on the top plot is used to fine-tune selections. May result in the bottom plot
                having inconsistent data display, in which case the annotations on the top plot will be correct with
                respect to the true (original) time values."""
            if event.artist != self.line1:
                return  # return if invalid pick target (not plot 1)
            if self.radio.value_selected == "None":
                return
            n = len(event.ind)
            if not n:
                return  # return if invalid pick target
            self.press = False  # avoid redrawing needlessly
            iloc_picked = event.ind[0]  # closest measurement picked (if multiple within pick radius)
            if self.radio.value_selected == "1 - Start":
                self.start_annotation.remove()
                self.start_point = self.df.iloc[iloc_picked]
                self.start_annotation = self.ax1.annotate(1, (self.start_point[0], self.start_point[1]),
                                                          fontsize=15)
            elif self.radio.value_selected == "2 - Breakdown":
                self.breakdown_annotation.remove()
                self.breakdown_point = self.df.iloc[iloc_picked]
                self.breakdown_annotation = self.ax1.annotate(2, (self.breakdown_point[0], self.breakdown_point[1]),
                                                              fontsize=15)
                self.isip_annotation.remove()
                self.isip_point = get_isip_point(self.df, self.breakdown_point)
                self.isip_annotation = self.ax1.annotate(3, (self.isip_point[0], self.isip_point[1]),
                                                         fontsize=15)
                self.closure_point = get_closure_point(self.df, self.breakdown_point, self.isip_point,
                                                       mode=self.radio2.value_selected)
                self.closure_annotation.remove()
                self.closure_annotation = self.ax1.annotate(4, (self.closure_point[0], self.closure_point[1]),
                                                            fontsize=15)

                self.used_df = self.df[self.isip_point.name:]  # for use in plot 2
                self.setup_plot2(mode=self.radio2.value_selected)

                self.line1_1.remove()
                self.line1_1, = self.ax1.plot((self.breakdown_point[0], self.isip_point[0]),
                                              (self.breakdown_point[1], self.isip_point[1]), color="tab:red")
            elif self.radio.value_selected == "3 - ISIP":
                self.isip_annotation.remove()
                self.isip_point = self.df.iloc[iloc_picked]
                self.isip_annotation = self.ax1.annotate(3, (self.isip_point[0], self.isip_point[1]),
                                                         fontsize=15)
                self.closure_point = get_closure_point(self.df, self.breakdown_point, self.isip_point,
                                                       mode=self.radio2.value_selected)
                self.closure_annotation.remove()
                self.closure_annotation = self.ax1.annotate(4, (self.closure_point[0], self.closure_point[1]),
                                                            fontsize=15)

                self.used_df = self.df[self.isip_point.name:]  # for use in plot 2
                self.setup_plot2(mode=self.radio2.value_selected)

                self.line1_1.remove()
                self.line1_1, = self.ax1.plot((self.breakdown_point[0], self.isip_point[0]),
                                              (self.breakdown_point[1], self.isip_point[1]), color="tab:red")
            elif self.radio.value_selected == "4 - Closure":
                self.closure_annotation.remove()
                self.closure_point = self.df.iloc[iloc_picked]
                self.closure_annotation = self.ax1.annotate(4, (self.closure_point[0], self.closure_point[1]),
                                                            fontsize=15)
                self.setup_plot2(mode=self.radio2.value_selected)
                # removed due to producing unexpected/inconsistent behaviour with plot 2
            plt.draw()
            return True

        def onclick(self, event):
            if event.inaxes != self.ax3:
                return  # return if invalid pick target (not plot 3) different approach necessary from pick event
            self.press = False  # avoid redrawing needlessly
            self.closure_annotation.remove()
            point_closeness_rank1 = (
                    self.used_df[self.radio2.value_selected] - event.xdata).abs().argsort()  # sorting by closeness
            point_closeness_rank2 = point_closeness_rank1[point_closeness_rank1 >= 0]  # eliminating null values
            self.closure_point = self.used_df.iloc[point_closeness_rank2.iloc[0] +
                                                   len(point_closeness_rank1) - len(
                point_closeness_rank2)]  # right index with respect to bottom ax
            closure_actual_index = (self.df.iloc[:, 0] - self.closure_point[0]).abs().argsort()[0]
            self.closure_point.name = closure_actual_index  # right index with respect to original df
            # very inefficient implementation!!!
            # left as is until major code rewrites are done, reindexing between dataframes is tedious
            self.closure_annotation = self.ax1.annotate(4, (self.closure_point[0], self.closure_point[1]),
                                                        fontsize=15)
            self.ax2_closure_marker.remove()
            self.ax2_closure_marker = self.ax3.axvline(self.closure_point[self.radio2.value_selected], color="black",
                                                       linestyle="--")
            plt.draw()

        def onpress(self, event):
            self.press = True

        def onmove(self, event):
            if self.press:
                self.move = True

        def onrelease(self, event):
            if self.press and not self.move:
                self.onclick(event)
            self.press = False
            self.move = False

        def on_xlims_change(self, event):
            if self.press:  # only on user input to avoid infinite recursion loops
                self.press = False  # avoids infinite recursion loops when xlims are updated on plot setup
                if event == self.ax1:
                    if self.radio.value_selected == "None":
                        return
                    limits = event.get_xlim()[0], event.get_xlim()[1]
                    df_aug = self.df[self.df.iloc[:, 0].between(*limits)].reset_index(drop=True)
                    index_offset = len(self.df[event.get_xlim()[0] >= self.df.iloc[:, 0]])
                    if self.radio.value_selected[0] == "1":
                        self.start_point = get_start_point(df_aug)
                        self.breakdown_point = get_breakdown_point(df_aug)
                        self.isip_point = get_isip_point(df_aug, self.breakdown_point)
                        self.closure_point = get_closure_point(df_aug, self.breakdown_point, self.isip_point)
                        self.start_point.name += index_offset
                        self.breakdown_point.name += index_offset
                        self.isip_point.name += index_offset
                        self.closure_point.name += index_offset
                        # all 4 points recalculated
                        try:
                            self.line1_1.remove()
                            self.start_annotation.remove()
                            self.breakdown_annotation.remove()
                            self.isip_annotation.remove()
                            self.closure_annotation.remove()
                        except ValueError:  # if the closure markers don't exist, for whatever reason, ignore them
                            pass
                        self.line1_1, = self.ax1.plot((self.breakdown_point[0], self.isip_point[0]),
                                                      (self.breakdown_point[1], self.isip_point[1]), color="tab:red")
                        self.start_annotation = self.ax1.annotate(1, (self.start_point[0], self.start_point[1]),
                                                                  fontsize=15)
                        self.breakdown_annotation = self.ax1.annotate(2, (
                        self.breakdown_point[0], self.breakdown_point[1]),
                                                                      fontsize=15)
                        self.isip_annotation = self.ax1.annotate(3, (self.isip_point[0], self.isip_point[1]),
                                                                 fontsize=15)
                        self.closure_annotation = self.ax1.annotate(4, (self.closure_point[0], self.closure_point[1]),
                                                                    fontsize=15)
                    elif self.radio.value_selected[0] == "2":
                        self.breakdown_point = get_breakdown_point(df_aug)
                        self.isip_point = get_isip_point(df_aug, self.breakdown_point)
                        self.closure_point = get_closure_point(df_aug, self.breakdown_point, self.isip_point)
                        self.breakdown_point.name += index_offset
                        self.isip_point.name += index_offset
                        self.closure_point.name += index_offset
                        # all except start point recalculated
                        try:
                            self.line1_1.remove()
                            self.breakdown_annotation.remove()
                            self.isip_annotation.remove()
                            self.closure_annotation.remove()
                        except ValueError:  # if the closure markers don't exist, for whatever reason, ignore them
                            pass
                        self.line1_1, = self.ax1.plot((self.breakdown_point[0], self.isip_point[0]),
                                                      (self.breakdown_point[1], self.isip_point[1]), color="tab:red")
                        self.breakdown_annotation = self.ax1.annotate(2, (
                        self.breakdown_point[0], self.breakdown_point[1]),
                                                                      fontsize=15)
                        self.isip_annotation = self.ax1.annotate(3, (self.isip_point[0], self.isip_point[1]),
                                                                 fontsize=15)
                        self.closure_annotation = self.ax1.annotate(4, (self.closure_point[0], self.closure_point[1]),
                                                                    fontsize=15)
                    elif self.radio.value_selected[0] == "3":
                        dummy_breakdown_point = get_breakdown_point(df_aug)
                        self.isip_point = get_isip_point(df_aug, dummy_breakdown_point)
                        self.closure_point = get_closure_point(df_aug, dummy_breakdown_point, self.isip_point)
                        self.isip_point.name += index_offset
                        self.closure_point.name += index_offset
                        # ISIP and closure points recalculated
                        try:
                            self.line1_1.remove()
                            self.isip_annotation.remove()
                            self.closure_annotation.remove()
                        except ValueError:  # if the closure markers don't exist, for whatever reason, ignore them
                            pass
                        self.line1_1, = self.ax1.plot((dummy_breakdown_point[0], self.isip_point[0]),
                                                      (dummy_breakdown_point[1], self.isip_point[1]), color="tab:red")
                        self.isip_annotation = self.ax1.annotate(3, (self.isip_point[0], self.isip_point[1]),
                                                                 fontsize=15)
                        self.closure_annotation = self.ax1.annotate(4, (self.closure_point[0], self.closure_point[1]),
                                                                    fontsize=15)
                    elif self.radio.value_selected[0] == "4":
                        # dummy_breakdown_point = get_breakdown_point(df_aug)
                        # dummy_isip_point = get_isip_point(df_aug, dummy_breakdown_point)
                        # self.closure_point = get_closure_point(df_aug, dummy_breakdown_point, dummy_isip_point)
                        # self.closure_point = get_closure_point(df_aug, self.breakdown_point, self.isip_point)
                        # using previously set breakdown and ISIP points instead of recalculated points because this is
                        # more likely to be intended by the user
                        # only closure point recalculated
                        self.closure_point = closure_point_outlier_removal(df_aug)
                        self.closure_point.name += index_offset
                        # sub-selecting the portion of the data to the find closure point in with
                        # closure_point_outlier_removal
                        try:
                            # self.line1_1.remove()
                            # no interaction with red line
                            self.closure_annotation.remove()
                        except ValueError:  # if the closure markers don't exist, for whatever reason, ignore them
                            pass
                        # self.line1_1, = self.ax1.plot((dummy_breakdown_point[0], dummy_isip_point[0]),
                        # (dummy_breakdown_point[1], dummy_isip_point[1]), color="tab:red")
                        # no interaction with red line
                        self.closure_annotation = self.ax1.annotate(4, (self.closure_point[0], self.closure_point[1]),
                                                                    fontsize=15)
                    self.used_df = df_aug
                    self.setup_plot2(mode=self.radio2.value_selected)
                elif event == self.ax2:
                    limits = event.get_xlim()[0], event.get_xlim()[1]
                    df_aug = self.used_df[self.used_df[self.radio2.value_selected].between(*limits)].reset_index(
                        drop=True)  # used_df instead of df so to reference correct limits
                    self.closure_point = closure_point_outlier_removal(df_aug, mode=self.radio2.value_selected)
                    self.closure_point.name += len(
                        self.df[event.get_xlim()[0] >= self.df[self.radio2.value_selected]]) + self.isip_point.name
                    # reindexing back to original dataframe
                    self.setup_plot1()
                    self.setup_plot2(mode=self.radio2.value_selected)
                    # self.ax2.autoscale_view()  # immobile view
                    # self.ax3.set_xlim(*limits)  # to make ax 3 zoom to ax2 correctly
                    # neither of the above seems to be necessary if x/y axis fixing is used
                    # self.ax2.set_xlim(*limits)
                    # self.ax3.set_xlim(*limits)
                    # this allows zooming on x axis, but can't use autoscale view without code rewrites
                    # default without scaling on zoom is good

        def show(self):
            plt.show()

    myplot = MyPlot()
    myplot.show()
    return myplot.df, myplot.start_point, myplot.breakdown_point, myplot.isip_point, myplot.closure_point


def add_slope_column(df):
    """Adds a smoothed pressure over time column. Expected input is a Pandas Dataframe that has cumulative time
            as its first column and pressure as its second column."""
    df["Slope"] = df.iloc[:, 1].diff() / df.iloc[:, 0].diff()  # slope without any smoothing/aggregation


def automatic_pas_file_processing(path):
    """Processes .pas files in the given format automatically, sensitive to deviations from format"""
    header_counter = 0
    header_line = 0  # line which denotes column labels
    data_rows = 0  # the number of row in the first dataset of the file
    with open(path, "r") as file:
        # hardcoded known values, a different approach may be more versatile
        for line in file:
            if "TCUM" in line and "PRGA" in line and header_line == 0:
                # only modify if both column names in line and not previously modified (to avoid second dataset)
                header_line = header_counter
            if "~DTG (1)" in line:
                data_rows += 1
            header_counter += 1
    try:
        pressure_test_df = pd.read_csv(path, header=header_line, nrows=data_rows - 1, usecols=["TCUM", "PRGA"])
    except UnicodeDecodeError:
        pressure_test_df = pd.read_csv(path, header=header_line, nrows=data_rows - 1, usecols=["TCUM", "PRGA"],
                                       encoding="cp1252")
    pressure_test_df = pressure_test_df.rename(columns={"TCUM": "Time (cumulative hours)", "PRGA": "Pressure (kPa)"})
    return pressure_test_df


def process_dfit_file(path):
    auto_parse = None
    while auto_parse not in {"Y", "N"}:
        auto_parse = input("Try parsing .pas file automatically [Y/N]: ").upper()
        if auto_parse == "Y":
            try:
                pressure_test_df = automatic_pas_file_processing(path)
                return pressure_test_df
            except Exception as e:
                print("Automatic parsing failed:", e, "\nDefaulting to manual selection")

    try:
        header = int(input("Enter the row number that contains the column names or leave blank for default "
                           "(the first one): ")) - 1
        # zero-indexed in Python, for user convenience using 1-indexing
    except ValueError:
        header = 0

    try:
        footer = int(input("Enter the number of rows to skip after the data (the number of footer rows) or leave blank "
                           "to process the entire file (if there are multiple datasets in the file the program will "
                           "attempt to use the first one automatically): "))
    except ValueError:
        footer = 0

    if os.path.splitext(path)[1] in {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls", ".xlt", ".xlm"}:
        # noinspection PyArgumentList
        sheet_name = input("Enter the name or number of the excel sheet with the DFIT data or leave blank for default "
                           "(the first one): ")
        try:
            sheet_name = int(sheet_name) - 1  # 1-indexing for user convenience
        except ValueError:
            pass
        if sheet_name == "":
            sheet_name = 0
        pressure_test_df = pd.read_excel(path, sheet_name=sheet_name, header=header, skipfooter=footer)
    else:  # currently only handling excel and comma separated files in the provided format
        try:
            pressure_test_df = pd.read_csv(path, header=header, skipfooter=footer)
        except UnicodeDecodeError:
            pressure_test_df = pd.read_csv(path, header=header, skipfooter=footer,
                                           encoding="cp1252")
        except pd.errors.ParserError:  # if the parser fails (in testing because there too many values in a row)
            separator_line = 0  # in files with two gauge datasets, line which denotes start of second dataset
            with open(path, "r") as file:
                for line in file:
                    separator_line += 1
                    if "~HEADER DATA - GAUGE (2)" in line:
                        # hardcoded known value at beginning of second dataset
                        # a different approach may be more versatile
                        break
            try:
                pressure_test_df = pd.read_csv(path, header=header, nrows=separator_line - header - 2)
            except UnicodeDecodeError:
                pressure_test_df = pd.read_csv(path, header=header, nrows=separator_line - header - 2,
                                               encoding="cp1252")

    col_select_string = "\n"
    for i, col in enumerate(pressure_test_df.columns):
        col_select_string += "\t[" + str(i + 1) + "] - " + col
    time_col = input(col_select_string + "\n" +
                     "Enter the column number that contains measurement time.\n"
                     "If there are columns for both date and cumulative time "
                     "(in hours with respect to the first measurement), "
                     "enter the column number that contains cumulative time.\n"
                     "If there are two columns separately containing date and time, "
                     "enter the column numbers separated by a comma "
                     "(e.g. if column 1 is MM/DD/YYYY and column 2 is HH:MM:SS enter 1, 2): ")
    try:
        time_col = int(time_col) - 1  # 1-indexing
    except ValueError:
        time_col = [int(x.strip(",")) - 1 for x in time_col.split(",")]
        pressure_test_df["Combined Date Time"] = pressure_test_df.iloc[:, time_col[0]] + " " + \
                                                 pressure_test_df.iloc[:, time_col[1]]
        time_col = pressure_test_df.columns.get_loc("Combined Date Time")

    pressure_col = int(input(col_select_string + "\n" +
                             "Enter the column number that contains measurement pressure (in kPa): ")) - 1  # 1-indexing
    pressure_test_df = pressure_test_df.iloc[:, [time_col, pressure_col]]
    pressure_test_df = pressure_test_df.rename(columns={pressure_test_df.columns[0]: "Time (cumulative hours)",
                                                        pressure_test_df.columns[1]: "Pressure (kPa)"})

    try:
        float(pressure_test_df["Time (cumulative hours)"][0])
        # if the conversion to float fails, then the data must be date time rather than cumulative time
    except ValueError:
        try:
            pressure_test_df["Time (cumulative hours)"] = pd.to_datetime(pressure_test_df["Time (cumulative hours)"]
                                                                         , format="%Y %m %d %H%M:%S")
            # known format first
        except ValueError:
            pressure_test_df["Time (cumulative hours)"] = pd.to_datetime(pressure_test_df["Time (cumulative hours)"]
                                                                         , infer_datetime_format=True)
            # try automatically determining the format if the known one fails
        pressure_test_df["Time (cumulative hours)"] = pressure_test_df["Time (cumulative hours)"]. \
            diff().dt.total_seconds().cumsum().div(3600).fillna(0)
        # cumulative time difference in hours

    return pressure_test_df


def main():
    app = QApplication(sys.argv)  # just to keep QApplication in memory, a gui event loop with exec_() isn't needed
    print("Select the file that contains the pressure test")
    dfit_path = QFileDialog().getOpenFileName(filter="All Files (*)")[0]
    if dfit_path == "":
        sys.exit()
    dfit_df = process_dfit_file(dfit_path)

    add_slope_column(dfit_df)

    start_point = get_start_point(dfit_df)

    breakdown_point = get_breakdown_point(dfit_df)

    isip_point = get_isip_point(dfit_df, breakdown_point)

    closure_point = get_closure_point(dfit_df, breakdown_point, isip_point)

    dfit_df, start_point, breakdown_point, isip_point, closure_point = plot_dfit(
        dfit_df, start_point, breakdown_point, isip_point, closure_point)
    # get points and df again from plotting function
    # getting df again is necessary for correct indexing with custom ISIP

    save_data_points(dfit_df, start_point, breakdown_point, isip_point, closure_point)

    pp_soliman, pp_nolte, pp_soliman_radial, pp_nolte_radial = plot_pore_pressure(
        dfit_df, start_point, isip_point, closure_point)

    save_pore_pressure(pp_soliman, pp_nolte, pp_soliman_radial, pp_nolte_radial)


if __name__ == "__main__":
    main()
