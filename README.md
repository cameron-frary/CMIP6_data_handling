This is a package written by Cameron Frary to make it simpler and easier to use data from the Coupled Model Intercomparison Project Phase 6 (CMIP6). There is only one object with seven methods. Users can easily select their desired model-experiment-variable combination, and create plots and animation. It is also easy to retrieve data from the object for external manipulation.

Using the package only requires cloning the repository and importing `CMIP6_Data_Manager` from `CMIP6_data_handling.data_handling`.

## Why

The object deals with all the initialization of objects necessary to the CMIP6 query, and deals with facets of plotting and animations that most would usually like to ignore.

## Drawbacks

The package has only been used in Google Colab. We had some difficulty installing and importing some dependencies on other systems and the code for doing so in Colab was provided.

Please also note that procedures involving comparisons between simulation data take significantly longer than the same procedures without comparison. Bottlenecks in the `xarray` package are to blame.

## Structure

### `CMIP6_Data_Manager`

This is the main object. It contains everything you need. The initialization function takes the following parameters:

- `query`: a dictionary containing fields to be passed to the CMIP6 search. Some common keys are `source_id` (model), `variable_id` (variable), `experiment_id` (experiment/forcing situation), and `table_id` (combination of part of model and time resolution). Values can be arrays. I highly recommend specifying at least the first three.
- `time_frame` (default `None`): a string specifying desired time resolution. Examples include `"mon"`, `"day"`, or `"yr"`. Not strictly necessary, but helpful when `table_id` is not specified in `query`.
- `specifics` (default `None`): a dictionary to specify additional dimensions, with the dimension name as the key and the desired value as the value (`{dim : val}`). Some variables will have a depth or altitude dimension (usually called `lev`). The dimension/key should be a string, and the val will usually be an integer.

It can take a long time to unpack the returned data, so I suggest limiting the number of variable/experiment combinations possible. 

In the following example usage, I query surface air pressure data (`ps`) from the GFDL-ESM4 model for the `historical` and `ssp585` experimental forcing situations. I want time resolution on the monthly scale, but I don't know the best table ID.
```
gfdl_esm4_pressure = CMIP6_Data_Manager(
    query= dict(
        source_id="GFDL-ESM4",
        variable_id=["ps"],
        experiment_id=["ssp585", "historical"]
    ),
    time_frame="mon"
)
```
We see from the following output that a `table_id` was automatically chosen. Other possible `table_id`s are displayed in case you want to specify a `table_id` (here `CFmon` might be a good option as it seems to be coupled---it incorporates data from several parts of the model). 
```
****Using Amon as table_id (options were ['Amon' 'CFday' 'CFmon' 'AERmon' 'Emon'])****
```



#### 
