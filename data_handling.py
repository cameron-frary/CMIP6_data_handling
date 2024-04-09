import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean
import pooch
import os
import tempfile
import gsw
import cv2

import time

tic = time.time()

import intake
import xarray as xr
import xesmf as xe

from xmip.preprocessing import combined_preprocessing
# from xarrayutils.plotting import shaded_line_plot

from datatree import DataTree
from xmip.postprocessing import _parse_metric

# @title Figure settings
import ipywidgets as widgets  # interactive display

plt.style.use(
    "https://raw.githubusercontent.com/ClimateMatchAcademy/course-content/main/cma.mplstyle"
)

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


def get_frame_data(query, time, col=col, kwargs=kwargs, lev=None):
  cat = col.search(
      **query,
      require_all_on=[
        "source_id"
      ],  # make sure that we only get models which have all of the above experiments
  )

  print(len(cat))

  if len(cat) > 1:
    print("Returned too many runs. Here is output:")
    print(f"source_ids: {cat.df['source_id'].unique()}")
    print(f"variable_ids: {cat.df['variable_id'].unique()}")
    print(f"member_ids: {cat.df['member_id'].unique()}")
    print(f"table_ids: {cat.df['table_id'].unique()}")
    print(f"grid_labels: {cat.df['grid_label'].unique()}")
    print(f"experiment_ids: {cat.df['experiment_id'].unique()}")
    raise Exception(f"Query returned {len(cat)} different runs")

  if len(cat) == 0:
    raise Exception(f"Query returned nothing")

  cat.esmcat.aggregation_control.groupby_attrs = ["source_id", "experiment_id"]
  dt = cat.to_datatree(**kwargs)

  data = dt[query["source_id"]][query["experiment_id"]].ds[query["variable_id"]]

  data_processed = data.sel(time=time).squeeze()

  if len(data_processed.dims) > 2:
    if lev is None:
      raise Exception(f"Too many dimensions: {data_processed.dims}. Specify more!")
    else:
      data_processed = data_processed.sel(lev=lev, method='nearest').squeeze()
      if len(data_processed.dims) > 2:
        raise Exception(f"Processed data with time and lev, still too many dimensions: {data_processed.dims}")

  return data_processed

def generate_map_plot(data, cmap, title, central_lon=0, vmin=None, vmax=None):
  fig, ax = plt.subplots(
    ncols=1, nrows=1, figsize = [8,4], subplot_kw={"projection": ccrs.PlateCarree(central_longitude=central_lon)}
  )

  try:
    p = data.plot(
      ax=ax,
      vmin=vmin,
      vmax=vmax,
      x="lon",
      y="lat",
      transform=ccrs.PlateCarree(),
      cmap=cmap,
      robust=True,
    )
  except ValueError:
    raise Exception("Use format 'cmocean.cm.[cmap name]' where cmap name is from https://matplotlib.org/cmocean/")

  ax.coastlines()
  ax.coastlines(color="grey", lw=0.5) #parameters for the lines on the coasts

  for lat in [-90, -60, -30, 0, 30, 60, 90]:
      ax.axhline(lat,color='k',ls='--')
  
  lon_formatter = LongitudeFormatter(zero_direction_label=True)
  ax.xaxis.set_major_formatter(lon_formatter)
  lat_formatter = LatitudeFormatter()
  ax.yaxis.set_major_formatter(lat_formatter)

  ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree(central_longitude=central_lon))
  ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())

  ax.add_feature(cart.feature.LAND, zorder=100, edgecolor="k")
  ax.set_title(title) # set a title

  return fig

def get_movie_data(query, time, col=col, kwargs=kwargs, lev=None):
  cat = col.search(
      **query,
      require_all_on=[
        "source_id"
      ],  # make sure that we only get models which have all of the above experiments
  )

  print(len(cat))

  if len(cat) > 1:
    raise Exception(f"Query returned {len(cat)} different runs")

  if len(cat) == 0:
    raise Exception(f"Query returned nothing")

  cat.esmcat.aggregation_control.groupby_attrs = ["source_id", "experiment_id"]
  dt = cat.to_datatree(**kwargs)

  data = dt[query["source_id"]][query["experiment_id"]].ds[query["variable_id"]]

  if len(data.dims) > 3:
    if lev is None:
      raise Exception(f"Too many dimensions: {data.dims}. Expected time, x, y")
    else:
      data_processed = data.sel(lev=lev, method='nearest').squeeze()
      if len(data_processed.dims) > 3:
        raise Exception(f"Processed data with time and lev, still too many dimensions: {data_processed.dims}")

  return data_processed

def make_movie(movie_data, years, month, cmap, vmin, vmax, central_lon=0, name="animation"):

  if "time" not in movie_data.dims:
    raise Exception(f"Attempted to make movie but missing time component: {movie_data.dims}")

  os.system("rm -r -f '/content/temp_images'")
  os.system("mkdir -p '/content/temp_images'")

  frames = []

  for year in years:  # years from parameters
    frame_data = movie_data.sel(time=f"{year}-{month}")
    p = generate_map_plot(
        data=frame_data,
        cmap=cmap,  # color mapping from parameters
        title=year,  # make sure to change title to what you want,
        central_lon=central_lon,
        vmin=vmin,
        vmax=vmax,
    )
    plt.close()
    p.savefig(f'/content/temp_images/{year}.png')
    frames.append(cv2.imread(f'/content/temp_images/{year}.png'))

  height,width,layers=frames[1].shape

  video=cv2.VideoWriter(
      f'/content/{name}.mp4',
      cv2.VideoWriter.fourcc(*"mp4v"),  # remind cameron to fix encoding from MJPG
      6, # fps
      (width,height)
  )

  for frame in frames:
      video.write(frame)

  cv2.destroyAllWindows()
  video.release()

class generator:
  def __init__(self, query, lev=None):
    self.query = query

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

    # if len(cat) > 1:
    #   print("Returned too many runs. Here is the output:")
    #   print("Returned too many runs. Here is output:")
    #   print(f"source_ids: {cat.df['source_id'].unique()}")
    #   print(f"variable_ids: {cat.df['variable_id'].unique()}")
    #   print(f"member_ids: {cat.df['member_id'].unique()}")
    #   print(f"table_ids: {cat.df['table_id'].unique()}")
    #   print(f"grid_labels: {cat.df['grid_label'].unique()}")
    #   print(f"experiment_ids: {cat.df['experiment_id'].unique()}")
    #   raise Exception(f"Query returned {len(cat)} different runs")

    if len(cat) == 0:
      raise Exception(f"Query returned nothing")

    cat.esmcat.aggregation_control.groupby_attrs = ["experiment_id","variable_id"]
    self.dt = cat.to_datatree(**kwargs)

    # self.data = dt[query["experiment_id"]][query["variable_id"]]
    # self.cmap = cmap
    self.lev = lev

  def set_lev(self, lev):
    self.lev = lev

  def get_data_frame(self, base, extra = None):
    # return self.dt[experiment][variable]

    time, experiment, variable = base
    time1 = None
    experiment1 = None
    variable1 = None
    if extra is not None:
      try:
        time1, experiment1, variable1 = extra
      except TypeError:
        raise Exception(f"Tried unpacking extra combo, but couldn't figure out how to unpack it. Expected (time, experiment, variable) format.")

    if len(time) == 7:
      data_processed = self.dt[experiment][variable].ds[variable].sel(time=time).squeeze()
    elif len(time) == 4:
      data_processed = self.dt[experiment][variable].ds[variable].sel(time=slice(f"{time}-01", f"{time}-12")).mean(dim="time").squeeze()
    else:
      raise Exception("Time input format matches neither 'YYYY' or 'YYYY-MM' (length {len(time)})") 

    # print(data_processed)

    if len(data_processed.dims) > 2:
      if self.lev is None:
        raise Exception(f"Too many dimensions: {data_processed.dims}. Specify more!")
      else:
        data_processed = data_processed.sel(lev=self.lev, method='nearest').squeeze()
        if len(data_processed.dims) > 2:
          raise Exception(f"Processed data with time and lev, still too many dimensions: {data_processed.dims}")
    
    if experiment1 is not None and variable1 is not None and time1 is not None:
      if len(time1) == 7:
        data_processed1 = self.dt[experiment1][variable1].ds[variable1].sel(time=time1).squeeze()
      elif len(time1) == 4:
        data_processed1 = self.dt[experiment1][variable1].ds[variable1].sel(time=slice(f"{time1}-01", f"{time1}-12")).mean(dim="time").squeeze()
      else:
        raise Expception("Time input format matches neither 'YYYY' or 'YYYY-MM' (length {len(time1)})") 

      if len(data_processed1.dims) > 2:
        if self.lev is None:
          raise Exception(f"Too many dimensions: {data_processed1.dims}. Specify more!")
        else:
          data_processed1 = data_processed1.sel(lev=self.lev, method='nearest').squeeze()
          if len(data_processed1.dims) > 2:
            raise Exception(f"Processed data with time and lev, still too many dimensions: {data_processed1.dims}")

      return data_processed1 - data_processed
    
    return data_processed

  def get_data_slides(self, experiment, variable):

    data_processed = self.dt[experiment][variable].ds[variable].squeeze()

    if len(data_processed.dims) > 3:
      if self.lev is None:
        raise Exception(f"Too many dimensions: {data_processed.dims}. Expected time, x, y")
      else:
        data_processed = self.dt[experiment][variable].sel(lev=self.lev, method='nearest').squeeze()
        if len(data_processed.dims) > 3:
          raise Exception(f"Processed data with time and lev, still too many dimensions: {data_processed.dims}")

    return data_processed

  def get_data_slides(self, base, extra=None):

    time, experiment, variable = base
    time1 = None
    experiment1 = None
    variable1 = None
    if extra is not None:
      try:
        time1, experiment1, variable1 = extra
      except TypeError:
        raise Exception(f"Tried unpacking extra combo, but couldn't figure out how to unpack it. Expected (experiment, variable) format.")

    data_processed = self.dt[experiment][variable].ds[variable].squeeze()

    if len(data_processed.dims) > 3:
      if self.lev is None:
        raise Exception(f"Too many dimensions: {data_processed.dims}. Expected time, x, y")
      else:
        data_processed = self.dt[experiment][variable].sel(lev=self.lev, method='nearest').squeeze()
        if len(data_processed.dims) > 3:
          raise Exception(f"Processed data with time and lev, still too many dimensions: {data_processed.dims}")

    if experiment1 is not None and variable1 is not None and time1 is not None:
      data_processed1 = self.dt[experiment1][variable1].ds[variable1].squeeze()

      if len(data_processed1.dims) > 3:
        if self.lev is None:
          raise Exception(f"Too many dimensions: {data_processed1.dims}. Expected time, x, y")
        else:
          data_processed1 = self.dt[experiment1][variable1].sel(lev=self.lev, method='nearest').squeeze()
          if len(data_processed1.dims) > 3:
            raise Exception(f"Processed data with time and lev, still too many dimensions: {data_processed1.dims}")

      if len(time) == 4:
        data_processed1 = data_processed1.sel(time=slice(f"{time}-01", f"{time}-12")).mean(dim="time")
      elif len(time) == 7:
        data_processed1 = data_processed1.sel(time=time)

      return data_processed1 - data_processed
    
    return data_processed

  def make_plot(self, cmap, title, base, extra=None, central_lon=0, vmin=None, vmax=None):
    
    fig, ax = plt.subplots(
      ncols=1, nrows=1, figsize = [8,4], subplot_kw={"projection": ccrs.PlateCarree(central_longitude=central_lon)}
    )

    data = self.get_data_frame(base, extra)
    
    try:
      p = data.plot(
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        robust=True,
      )
    except ValueError:
      raise Exception("Use format 'cmocean.cm.[cmap name]' where cmap name is from https://matplotlib.org/cmocean/")

    ax.coastlines()
    ax.coastlines(color="grey", lw=0.5) #parameters for the lines on the coasts
      
    for lat in [-90, -60, -30, 0, 30, 60, 90]:
      ax.axhline(lat,color='k',ls='--')
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
  
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree(central_longitude=central_lon))
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())

    ax.add_feature(cart.feature.LAND, zorder=100, edgecolor="k")
    ax.set_title(title) # set a title

    return fig

  def make_animation(self, years, months, vmin, vmax, cmap, base, extra=None, central_lon=0, name="animation"):
    movie_data = self.get_data_slides(base, extra)
    
    if "time" not in movie_data.dims:
      raise Exception(f"Attempted to make movie but missing time component: {movie_data.dims}")

    #!rm -r -f "/content/temp_images"
    #!mkdir -p "/content/temp_images"
    os.system("rm -r -f '/content/temp_images'")
    os.system("mkdir -p '/content/temp_images'")

    frames = []
    
    for year in years:  # years from parameters
      if len(months) == 0:
        frame_data = movie_data.sel(time=slice(f"{year}-01", f"{year}-12")).mean(dim="time")
        p = generate_map_plot(
            data=frame_data,
            cmap=cmap,  # color mapping from parameters
            title=f"Average for {year}",  # make sure to change title to what you want,
            central_lon=central_lon,
            vmin=vmin,
            vmax=vmax
        )
        plt.close()
        p.savefig(f'/content/temp_images/{year}.png')
        frames.append(cv2.imread(f'/content/temp_images/{year}.png'))

      else:
        for month in months:
          print(f"{year}-{month:02d}")
          frame_data = movie_data.sel(time=f"{year}-{month:02d}")    
          p = generate_map_plot(
              data=frame_data,
              cmap=cmap,  # color mapping from parameters
              title=f"{year}-{month:02d}",  # make sure to change title to what you want,
              central_lon=central_lon,
              vmin=vmin,
              vmax=vmax
          )
          plt.close()
          p.savefig(f'/content/temp_images/{year}-{month:02d}.png')
          frames.append(cv2.imread(f'/content/temp_images/{year}-{month:02d}.png'))

    height,width,layers=frames[1].shape

    video=cv2.VideoWriter(
        f'/content/{name}.mp4',
        cv2.VideoWriter.fourcc(*"mp4v"),
        6, # fps
        (width,height)
    )

    for frame in frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

