This is a package written by Cameron Frary to make it simpler and easier to use data from the Coupled Model Intercomparison Project Phase 6 (CMIP6). There is only one object with seven methods. Users can easily select their desired model-experiment-variable combination, and create plots and animation. It is also easy to retrieve data from the object for external manipulation.

Using the package only requires cloning the repository and importing `CMIP6_Data_Manager` from `CMIP6_data_handling.data_handling`.

## Why

The object deals with all the initialization of objects necessary to the CMIP6 query, and deals with facets of plotting and animations that most would usually like to ignore.

## Drawbacks

The package has only been used in Google Colab. We had some difficulty installing and importing some dependencies on other systems and the code for doing so in Colab was provided.

Please also note that procedures involving comparisons between simulation data take significantly longer than the same procedures without comparison. Bottlenecks in the `xarray` package are to blame.

## Structure

### `CMIP6_Data_Manager`

This is the main object.
