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

def generate_map_plot(data, cmap, title, vmin=None, vmax=None):
  fig, ax = plt.subplots(
    ncols=1, nrows=1, figsize = [8,4], subplot_kw={"projection": ccrs.PlateCarree()}
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
    
  ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
  ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
  
  lon_formatter = LongitudeFormatter(zero_direction_label=True)
  lat_formatter = LatitudeFormatter()

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

def make_movie(movie_data, years, month, cmap, vmin, vmax, name="animation"):

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
        vmin=vmin,
        vmax=vmax
    )
    plt.close()
    p.savefig(f'/content/temp_images/{year}.png')
    frames.append(cv2.imread(f'/content/temp_images/{year}.png'))

  height,width,layers=frames[1].shape

  video=cv2.VideoWriter(
      f'/content/{name}.mp4',
      cv2.VideoWriter.fourcc(*"mp4v"),  # remind cameron to fix encoding from MJPG
      24, # fps
      (width,height)
  )

  for frame in frames:
      video.write(frame)

  cv2.destroyAllWindows()
  video.release()

class generator:
  def __init__(self, query, cmap, lev=None):
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

    if len(cat) > 1:
      print("Returned too many runs. Here is the output:")
      raise Exception(f"Query returned {len(cat)} different runs")

    if len(cat) == 0:
      raise Exception(f"Query returned nothing")

    cat.esmcat.aggregation_control.groupby_attrs = ["source_id", "experiment_id"]
    dt = cat.to_datatree(**kwargs)

    self.data = dt[query["source_id"]][query["experiment_id"]].ds[query["variable_id"]]
    self.cmap = cmap
    self.lev = lev

  def set_lev(self, lev):
    self.lev = lev

  def get_data_frame(self, time):
    data_processed = self.data.sel(time=time).squeeze()

    if len(data_processed.dims) > 2:
      if self.lev is None:
        raise Exception(f"Too many dimensions: {data_processed.dims}. Specify more!")
      else:
        data_processed = data_processed.sel(lev=self.lev, method='nearest').squeeze()
        if len(data_processed.dims) > 2:
          raise Exception(f"Processed data with time and lev, still too many dimensions: {data_processed.dims}")

    return data_processed

  def get_data_slides(self):

    data_processed = self.data

    if len(data_processed.dims) > 3:
      if self.lev is None:
        raise Exception(f"Too many dimensions: {data_processed.dims}. Expected time, x, y")
      else:
        data_processed = self.data.sel(lev=self.lev, method='nearest').squeeze()
        if len(data_processed.dims) > 3:
          raise Exception(f"Processed data with time and lev, still too many dimensions: {data_processed.dims}")

    return data_processed

  def make_plot(self, time, title, vmin=None, vmax=None):
    fig, ax = plt.subplots(
      ncols=1, nrows=1, figsize = [8,4], subplot_kw={"projection": ccrs.PlateCarree()}
    )

    data = self.get_data_frame(time)
    try:
      p = data.plot(
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap=self.cmap,
        robust=True,
      )
    except ValueError:
      raise Exception("Use format 'cmocean.cm.[cmap name]' where cmap name is from https://matplotlib.org/cmocean/")

    ax.coastlines()
    ax.coastlines(color="grey", lw=0.5) #parameters for the lines on the coasts

    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
      
    for lat in [-90, -60, -30, 0, 30, 60, 90]:
      ax.axhline(lat,color='k',ls='--')
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()

    ax.add_feature(cart.feature.LAND, zorder=100, edgecolor="k")
    ax.set_title(title) # set a title

    return fig

  def make_animation(self, years, month, vmin, vmax, name="animation"):
    movie_data = self.get_data_slides()

    if "time" not in movie_data.dims:
      raise Exception(f"Attempted to make movie but missing time component: {movie_data.dims}")

    #!rm -r -f "/content/temp_images"
    #!mkdir -p "/content/temp_images"
    os.system("rm -r -f '/content/temp_images'")
    os.system("mkdir -p '/content/temp_images'")

    frames = []

    for year in years:  # years from parameters
      frame_data = movie_data.sel(time=f"{year}-{month}")
      p = generate_map_plot(
          data=frame_data,
          cmap=self.cmap,  # color mapping from parameters
          title=year,  # make sure to change title to what you want,
          vmin=vmin,
          vmax=vmax
      )
      plt.close()
      p.savefig(f'/content/temp_images/{year}.png')
      frames.append(cv2.imread(f'/content/temp_images/{year}.png'))

    height,width,layers=frames[1].shape

    video=cv2.VideoWriter(
        f'/content/{name}.mp4',
        cv2.VideoWriter.fourcc(*"mp4v"),
        24, # fps
        (width,height)
    )

    for frame in frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

