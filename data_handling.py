import matplotlib.pyplot as plt
import os
import cv2

import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import intake
from xmip.preprocessing import combined_preprocessing


class CMIP6_Data_Manager:
    def __init__(self, query, time_frame=None, specifics=None):
        if specifics is None:
            specifics = {}
        self.query = query
        self.baseline_params = (None, None, None)
        self.baseline_data = None

        plt.style.use(
            "https://raw.githubusercontent.com/ClimateMatchAcademy/course-content/main/cma.mplstyle"
        )

        # %matplotlib inline

        col = intake.open_esm_datastore(
            "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        )

        kwargs = dict(
            preprocess=combined_preprocessing,  # apply xMIP fixes to each dataset
            xarray_open_kwargs=dict(
                use_cftime=True
            ),  # ensure all datasets use the same time index
            storage_options={
                "token": "anon"
            },  # anonymous/public authentication to google cloud storage
        )

        cat = col.search(
            **self.query,
            require_all_on=[
                "source_id"
            ],  # make sure that we only get models which have all of the above experiments
        )

        if len(cat) == 0:
            raise Exception(f"Query returned nothing")

        if "table_id" not in query:
            if time_frame is None:
                raise Exception(
                    "No table_id or time_frame specified. Either specify table_id in query, or time_frame as general "
                    "parameter.")

            table_id_found = False

            # Gets first table_id that includes the time_frame, adds it to query, and re-runs query
            # Raises an error if no appropriate table_id found
            for possible_id in cat.df["table_id"].unique():
                if time_frame in possible_id:
                    print(f"****Using {possible_id} as table_id (options were {cat.df['table_id'].unique()})****")
                    table_id_found = True
                    query["table_id"] = possible_id

                    cat = col.search(
                        **self.query,
                        require_all_on=[
                            "source_id"
                        ],  # make sure that we only get models which have all of the above experiments
                    )

                    break

            if not table_id_found:
                raise Exception(
                    f"No table_id found with time frame '{time_frame}'. "
                    f"Possible table_ids: {cat.df['table_id'].unique()}")

        self.cat = cat.esmcat.aggregation_control.groupby_attrs = ["experiment_id", "variable_id"]
        print(f"source_ids: {cat.df["table_id"].unique()}")
        print(f"experiment_ids: {cat.df["experiment_id"].unique()}")
        print(f"member_ids: {cat.df["member_id"].unique()}")
        self.dt = cat.to_datatree(**kwargs)

        # self.data = dt[query["experiment_id"]][query["variable_id"]]
        # self.cmap = cmap
        self.specifics = specifics

    def sel_time(self, data, experiment, time, data_slice=False):

        # If experiment is one of these two historical experiments, get average over
        # all years. Else, average over interval if given, or select specific time
        if not data_slice and (experiment == "historical" or experiment == "land-hist"):
            data = data.mean(dim="time").squeeze()
        elif len(time) == 1:
            data = data.sel(time=time[0]).squeeze()
        elif not data_slice and len(time) == 2:
            data = data.sel(time=slice(time[0], time[1])).mean(dim="time").squeeze()
        elif data_slice and len(time) == 2:
            data = data.sel(time=slice(time[0], time[1])).squeeze()
        else:
            raise Exception(
                "Time input format is neither ('YYYY-MM') or ('YYYY-MM', 'YYYY-MM'), but can be empty when experiment "
                "is historical or land-hist")

        return data

    def attempt_dim_fix(self, data, desired_dims):
        while len(data.dims) > desired_dims:
            print(f"Need to specify more dimensions of set {data.dims}! Aiming for {desired_dims} free dimensions.")
            dimension = input("Which dimension do you want to specify? ")
            value = int(input(f"What value do you want to use for {dimension}"))

            try:
                self.specifics[dimension] = value
                print(self.specifics)
                data = data.sel(self.specifics, method='nearest').squeeze()
            except KeyError:
                print(
                    "That didn't work. Try again or key interrupt. Maybe specify more specifics with "
                    "'self.specifics[dim] = val'?")

        return data

    def sel_specifics(self, data, experiment):

        # If baseline variable is primary productivity, then sum up values from 0m to 100m,
        # and multiply by 100m (basically Riemann sum assuming lev intervals are even)
        if experiment == "pp":
            print("WARNING: PRIMARY PRODUCTIVITY IS AUTOMATICALLY INTEGRATED OVER DEPTH")
            data = data.sel(lev=slice(0, 100), method='nearest').sum(dim="lev").squeeze() * 100
        else:
            data = data.sel(self.specifics, method='nearest').squeeze()

        return data

    def update_baseline(self, baseline_params):

        print(f"Updating baseline with params {baseline_params}")

        # Save and unpack baseline parameters
        self.baseline_params = baseline_params
        try:
            base_time, base_experiment, base_variable = baseline_params
        except TypeError:
            raise Exception(
                f"Tried unpacking extra combo, but couldn't figure out how to unpack it. Expected (time, experiment, "
                f"variable) format.")

        # Fetch the basic data for the experiment and variable
        self.baseline_data = self.dt[base_experiment][base_variable].ds[base_variable]

        # Select specifics like lev, also deal with cases like experiment = primary productivity integrating over depth
        self.baseline_data = self.sel_specifics(self.baseline_data, base_experiment)

        # Select time, averaging over all historical if experiment = historical
        self.baseline_data = self.sel_time(self.baseline_data, base_experiment, base_time)

        # If there are still outstanding dimensions, try to fix. If fix fails, give suggestions
        self.baseline_data = self.attempt_dim_fix(self.baseline_data, desired_dims=2)

    def get_data(self, main, baseline_params=None, data_slice=False):
        # return self.dt[experiment][variable]

        # If there are baseline parameters that don't match what's already in there, update baseline
        # If no baseline is given, then remove baseline parameters and data
        if baseline_params is not None and baseline_params != self.baseline_params:
            self.update_baseline(baseline_params)
        elif baseline_params is None:
            self.baseline_params = None
            self.baseline_data = None

        # Unpack main time, experiment, and variable
        try:
            time, experiment, variable = main
        except KeyError:
            raise Exception(
                "Was expecting main input to have one of the following formats:\n  ([time_start, time_end], "
                "experiment, variable)\n  ([time], experiment, variable)")

        # Select basic main data
        main_data = self.dt[experiment][variable].ds[variable]

        # Select appropriate time, specifics (like lev), and attempt to fix too many dimensions issue
        main_data = self.sel_time(main_data, experiment, time, data_slice)
        main_data = self.sel_specifics(main_data, experiment)

        main_data = self.attempt_dim_fix(
            main_data,
            desired_dims=3 if data_slice else 2
        )

        # If there is baseline data, then return change as multiple of historical, otherwise just main
        if self.baseline_data is not None:
            return (main_data - self.baseline_data) / self.baseline_data
        else:
            return main_data

    def make_plot(self, title, cmap_label, cmap, main, baseline=None, central_lon=0, vmin=None, vmax=None,
                  block_land=True):

        # Initialize figure (bit)
        fig, ax = plt.subplots(
            ncols=1, nrows=1, figsize=[12, 6],
            subplot_kw={"projection": ccrs.PlateCarree(central_longitude=central_lon)}
        )

        # Get data
        data = self.get_data(main, baseline)

        # Plot (alerting if cmap error)
        try:
            # with Profile() as profile:
            data.plot(
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                x="lon",
                y="lat",
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                robust=True,
                cbar_kwargs={"label": cmap_label}
            )
            # Stats(profile).strip_dirs().sort_stats("cumtime").print_stats()
        except ValueError:
            # This may be misleading and cause problems if the value error is from something besides cmap.
            raise Exception(
                "Use format 'cmocean.cm.[cmap name]' where cmap name is from https://matplotlib.org/cmocean/")
        except TypeError:
            raise Exception(f"Pretty sure there are too many dimensions: {data.dims}")

        # If land should be blocked (i.e. looking only at ocean), draw land in
        if block_land:
            ax.add_feature(cart.feature.LAND, zorder=100, edgecolor="k")
        ax.coastlines()
        ax.coastlines(color="k", lw=0.75)  # parameters for the lines on the coasts

        # Add lat lines (nice for looking at atmos cells)
        for lat in [-90, -60, -30, 0, 30, 60, 90]:
            ax.axhline(lat, color='k', ls='--')

        # Format lat and lon tick marks
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        ax.xaxis.set_major_formatter(lon_formatter)
        lat_formatter = LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)

        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree(central_longitude=central_lon))
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())

        ax.set_title(title)  # set a title

        return fig

    def make_animation(self, years, months, vmin, vmax, cmap, main, cmap_label, baseline=None, central_lon=0,
                       name="animation"):
        # Remove and create temp_images folder (to clear folder)
        os.system("rm -r -f '/content/temp_images'")
        os.system("mkdir -p '/content/temp_images'")

        # Initialize array for OpenCV VideoWriter
        frames = []

        # Unpack main time, experiment, and variable
        main_time, main_exp, main_var = main

        print(f"Years: {len(years)}, months: {len(months)}")
        times = []

        # For year in years....
        for year in years:
            # Add times
            if len(months) == 0:
                times.append([f"{year}-01", f"{year}-12"])
            else:
                for month in months:
                    times.append([f"{year}-{month:02d}"])

        for time in times:
            if len(time) == 2:
                year = time[0][0:4]
                title = f"Average for {year}"  # make sure to change title to what you want
            elif len(time) == 1:
                year_month = time[0]
                title = f"{year_month}"  # make sure to change title to what you want
            else:
                raise Exception(f"Time given ({time}) is neither an interval or single value")

            print(title)
            p = self.make_plot(
                cmap=cmap,  # color mapping from parameters
                title=title,
                main=(time, main_exp, main_var),
                baseline=baseline,
                central_lon=central_lon,
                vmin=vmin,
                vmax=vmax,
                cmap_label=cmap_label
            )
            plt.close()
            p.savefig(f'/content/temp_images/{title}.png')
            frames.append(cv2.imread(f'/content/temp_images/{title}.png'))

        height, width, layers = frames[1].shape

        video = cv2.VideoWriter(
            f'/content/{name}.mp4',
            cv2.VideoWriter.fourcc(*"mp4v"),
            6,  # fps
            (width, height)
        )

        for frame in frames:
            video.write(frame)

        cv2.destroyAllWindows()
        video.release()
