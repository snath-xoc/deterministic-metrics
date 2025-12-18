# Functions to load daily mean data

import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import h5py
from datetime import datetime, timedelta
import xesmf as xe
import xarray as xr
import os.path

Ethiopia_mask_file_name = "/Users/cooperf/Documents/WFP/data/IMERG/ICPAC_region/Ethiopia_state_masks_for_IMERG.nc"
Kenya_mask_file_name = "/Users/cooperf/Documents/WFP/data/IMERG/ICPAC_region/Kenya_county_masks_for_IMERG.nc"
country_masks_file_name = "/Users/cooperf/Documents/WFP/data/IMERG/ICPAC_region/ICPAC_country_masks_for_IMERG.nc"
# XXX Probably not needed here
ICPAC_climate_data_file = "/Users/cooperf/Documents/WFP/data/WRF_ICPAC_1981-2010-monthly.nc"


# XXX Move to utility functions
# Returns the number of days in a given month
def num_days_in_month(year,month):
    if (month < 12):
        days_this_month = 1#(datetime(year, month+1, 1) - datetime(year, month, 2)).days
    else:
        days_this_month = (datetime(year+1, 1, 1) - datetime(year, month, 1)).days
    return days_this_month


# Load the country mask
# Arguments:
#    country_to_look_at - "South Sudan","Rwanda","Burundi","Djibouti","Eritrea",
#                         "Ethiopia","Sudan","Somalia","Tanzania","Kenya","Uganda"
# Returns:
#    latitude           - An array of the latitudes in the mask
#    longitude          - An array of the longitudes in the mask
#    mask               - An array of booleans of size (len(latitude),len(longitude))
#                         corresponding to the country mask
def load_country_mask(country_to_look_at):
    # XXX Replace latitude, longitude, mask with the country name.
    # XXX Add the ICPAC region as a location
    
    # Load the country mask
    if (country_to_look_at == "Kenya"):
        # The Kenya mask computed from the KMD shapefile
        nc_file = nc.Dataset(Kenya_mask_file_name)
    elif (country_to_look_at == "Ethiopia"):
        # The Ethiopia mask computed from the KMD shapefile
        nc_file = nc.Dataset(Ethiopia_mask_file_name)
    else:
        nc_file = nc.Dataset(country_masks_file_name)

    latitude = np.array(nc_file[f"latitude_{country_to_look_at}"][:])
    longitude = np.array(nc_file[f"longitude_{country_to_look_at}"][:])
    # True when masked, False when valid.
    mask = np.logical_not(np.moveaxis(np.array(nc_file[f"mask_{country_to_look_at}"][:], dtype=bool),0,1))
    nc_file.close()
    
    return latitude, longitude, mask


# Load and compute the monthly climatology from ICPAC
# XXX Is the ICPAC climatology in mm/h and not mm/day as the NetCDF file suggests?
# XXX Or has it been summed over the month or what?
def load_ICPAC_monthly_climatology(ICPAC_climate_data_file,
                                   region="Kenya"
                                  ):
    
    # Which country are we looking at. Can be:
    # "South Sudan","Rwanda","Burundi","Djibouti","Eritrea",
    # "Ethiopia","Sudan","Somalia","Tanzania","Kenya","Uganda"
    latitude, longitude, mask = load_country_mask(region)
    
    # Load the monthly mean ICPAC-WRF data
    nc_file = nc.Dataset(ICPAC_climate_data_file)
    latitude_ICPAC = np.array(nc_file["lat"][:])
    longitude_ICPAC = np.array(nc_file["lon"][:])
    # time_ICPAC = np.array(nc_file["Time"][:])
    precip_ICPAC = np.array(nc_file["prec"][:])
    nc_file.close()
    
    # Take the 30 year average (starting 1981) to get the monthly climatology
    precip_clim_ICPAC_np = np.mean(np.reshape(precip_ICPAC[0:360,:,:],
                                           (30,12,len(latitude_ICPAC),len(longitude_ICPAC))
                                          ), axis=0)

    # Create an xarray DataArray for the latitudes
    latitude_ICPAC_da = xr.DataArray(latitude_ICPAC,
                                     dims="lat",
                                     attrs={"units": "degrees_north"})

    # Create an xarray DataArray for the longitudes
    longitude_ICPAC_da = xr.DataArray(longitude_ICPAC,
                                     dims="lon",
                                     attrs={"units": "degrees_east"})

    # Create an xarray DataArray for interpolation
    precip_clim_ICPAC_da = xr.DataArray(precip_clim_ICPAC_np,
                                        dims=("month","lat","lon"),
                                        coords={"month": np.arange(12)+1,
                                                "lat": latitude_ICPAC_da,
                                                "lon": longitude_ICPAC_da},
                                        attrs={"units": "mm/day"})

    # Place the DataArray into a Dataset
    ds_in = xr.Dataset({"prec": precip_clim_ICPAC_da})
    
    if (latitude is None) and (longitude is None):
        
        # No interpolation required
        dr_out = ds_in["prec"]
    
    else:  # Interpolate to the new grid
        
        # Specify the latitude and longitude to interpolate to if not specified
        if (latitude is None):
            latitude = latitude_ICPAC
        if (longitude is None):
            longitude = longitude_ICPAC
        
        # Create the Dataset specification we are interpolating to
        ds_out = xr.Dataset({"lat": (["lat"], latitude, {"units": "degrees_north"}),
                             "lon": (["lon"], longitude, {"units": "degrees_east"})})

        # Define the regridder function to perform the interpolation
        regridder = xe.Regridder(ds_in, ds_out, "conservative")

        # Regrid the DataArray
        dr_out = regridder(ds_in["prec"], keep_attrs=True)

    precip_clim_ICPAC = dr_out
        
    if (mask is not None):  # Apply the mask
        
        # Repeat the mask for each month
        annual_mask = np.repeat(np.reshape(mask, (1,len(latitude),len(longitude))), 12, axis=0)

        # Mask the rainfall to the chosen region
        precip_clim_ICPAC.values = ma.array(precip_clim_ICPAC.values, mask = annual_mask)
    
    # Returns an xarray DataArray (nan's in the mask region)
    # XXX Is the ICPAC climatology in mm/h and not mm/day as the NetCDF file suggests?
    # XXX Or has it been summed over the month or what?
    for month in range(1,13):
        precip_clim_ICPAC[month-1,:,:] /= num_days_in_month(2023,month)

    return precip_clim_ICPAC


# Load daily mean IMERG data for one month (returns mm/day)
def load_daily_mean_IMERG_by_month(year,                # Year to load
                                   month,               # Month to load (1-12)
                                   data_dir,            # Where the data is stored
                                   lead_time_offset=0,  # Hours since 00:00 UTC before the first lead time
                                   latitude=None,       # List of latitudes to load (None means load all)
                                   longitude=None,      # List of longitudes to load (None means load all)
                                   IMERG_version=7,     # Version of IMERG data to load (6 or 7)
                                   mask=None            # A mask: True where data is missing, False otherwise
                                  ):
    # Start at 00:00 UTC + lead_time_offset in hours
    d_start = datetime(year,month,1) + timedelta(hours=lead_time_offset)
    d = d_start
    
    if (IMERG_version == 7):
        precip_data_name = "precipitation"
        name_string_p1 = "3B-HHR.MS.MRG.3IMERG."
        name_string_p2 = ".V07B.HDF5"
    elif (IMERG_version == 6):
        precip_data_name = "precipitationCal"
        name_string_p1 = "3B-HHR-L.MS.MRG.3IMERG."  # Late run
        if (d < datetime(2023,7,1,14)):
            name_string_p2 = ".V06C.HDF5"
        elif (d < datetime(2023,11,9,2)):
            name_string_p2 = ".V06D.HDF5"
        else:
            name_string_p2 = ".V06E.HDF5"
    else:
        print(f"WARNING: Only tested with IMERG v6 or v7 files but v{IMERG_version} specified.")
        precip_data_name = "precipitation"  # Use this name and continue
        name_string_p1 = "3B-HHR.MS.MRG.3IMERG."
        name_string_p2 = f".V0{IMERG_version}A.HDF5"

    # Load full latitude and longitude from the first file
    d2 = d + timedelta(seconds=30*60-1)
    file_name = f"{data_dir}/{d.year}/{d.strftime('%b')}/{name_string_p1}{d.year}{d.month:02d}{d.day:02d}-S{d.hour:02d}{d.minute:02d}00-E{d2.hour:02d}{d2.minute:02d}59.{d.hour*60+d.minute:04d}{name_string_p2}"
    h5_file = h5py.File(file_name)
    latitude_full = h5_file['Grid']['lat'][:]
    longitude_full = h5_file['Grid']['lon'][:]
    h5_file.close()
    
    # Find the latitude indices specifing the region to load
    if (latitude is None):
        latitude = latitude_full
        min_lat_idx = None
        max_lat_idx = None
    else:
        min_lat_idx = np.argmin(np.abs(latitude_full - np.min(latitude)))
        max_lat_idx = np.argmin(np.abs(latitude_full - np.max(latitude))) + 1
    
    # Find the longitude indices specifing the region to load
    if (longitude is None):
        longitude = longitude_full
        min_lon_idx = None
        max_lon_idx = None
    else:
        min_lon_idx = np.argmin(np.abs(longitude_full - np.min(longitude)))
        max_lon_idx = np.argmin(np.abs(longitude_full - np.max(longitude))) + 1
    
    # Daily precipitation for the month
    days_this_month = num_days_in_month(year,month)
    daily_precip_IMERG = np.zeros((days_this_month, len(longitude), len(latitude)))

    # While we are still looking at this month's data
    day_idx = 0
    d_end_month = d_start + timedelta(days=days_this_month)
    while (d < d_end_month):

        # Precipitation for the day every 30 mins
        precipitation = np.zeros((len(longitude), len(latitude)))

        # For the current day
        d_end = d + timedelta(hours=24)
        while (d < d_end):

            # The IMERGv6 version changes at certain times throughout the year
            if (IMERG_version == 6):
                if (d < datetime(2023,7,1,14)):
                    name_string_p2 = ".V06C.HDF5"
                elif (d < datetime(2023,11,8,2)):
                    name_string_p2 = ".V06D.HDF5"
                else:
                    name_string_p2 = ".V06E.HDF5"
            
            # Load a single IMERG data file cropping to the bounding box
            d2 = d + timedelta(seconds=30*60-1)
            file_name = f"{data_dir}/{d.year}/{d.strftime('%b')}/{name_string_p1}{d.year}{d.month:02d}{d.day:02d}-S{d.hour:02d}{d.minute:02d}00-E{d2.hour:02d}{d2.minute:02d}59.{d.hour*60+d.minute:04d}{name_string_p2}"
            h5_file = h5py.File(file_name)
            latitude_IMERG = h5_file['Grid']['lat'][min_lat_idx:max_lat_idx]
            longitude_IMERG = h5_file['Grid']['lon'][min_lon_idx:max_lon_idx]
            time = h5_file['Grid']['time'][:]
            # There is only one time, so remove this axis with 0
            precipitation += h5_file['Grid'][precip_data_name][0,min_lon_idx:max_lon_idx,min_lat_idx:max_lat_idx]
            h5_file.close()

            # Check that latitude and longitude are correct
            if (np.max(np.abs(latitude - latitude_IMERG)) > 1e-5):
                print("ERROR: latitude to load and lattiude in IMERG data are not equal.")
            if (np.max(np.abs(longitude - longitude_IMERG)) > 1e-5):
                print("ERROR: longitude to load and longitude in IMERG data are not equal.")

            # Check that the date is correct
            if (IMERG_version == 6):
                if (d != datetime(1970,1,1) + timedelta(seconds=int(time[0]))):
                    print(f"ERROR: Date not correct in {file_name}")
            elif (IMERG_version == 7):
                if (d != datetime(1980,1,6) + timedelta(seconds=int(time[0]))):
                    print(f"ERROR: Date not correct in {file_name}")
            # Date not checked with other versions of IMERG
            
            # Move to the next data file
            d += timedelta(minutes=30)

        # Compute the daily mean precipitation in mm/day
        daily_precip_IMERG[day_idx,:,:] = precipitation * 24 / 48
        day_idx += 1

    # Dimensions need flipping
    daily_precip_IMERG = np.moveaxis(daily_precip_IMERG,1,2)
    
    if mask is not None:
        # Create the correct size mask for the IMERG data
        daily_precip_IMERG_mask = np.repeat(
            np.reshape(mask,(1,len(latitude),len(longitude))),
            days_this_month, axis=0)

        # Create a masked array
        daily_precip_IMERG = ma.masked_array(daily_precip_IMERG, mask=daily_precip_IMERG_mask)
    
    # XXX Should probably return an xarray dataset instead
    return daily_precip_IMERG


# XXX What about Janurary with the time offsets?
# XXX Deal with time offsets at the start of the file
# XXX There is a problem with the times in the masked IMERG data files
# XXX Specify the units in the masked IMERG file
def load_daily_mean_IMERG_by_month_simplified(
    year,             # Year to load
    month,            # Month to load (1-12)
    data_dir,         # Where the data is stored
    mask=None         # A mask: True where data is missing, False otherwise
):
    
    # Load the IMERG data for this month
    files = f"{data_dir}/{year}/{year}{month:02d}*.nc"
    nc_file = xr.open_mfdataset(files).sortby('time')
    latitude_IMERG = nc_file.latitude.values
    longitude_IMERG = nc_file.longitude.values
    time_IMERG = nc_file.time.values[1:-3].reshape(-1,4)[:,0]
    daily_precip_IMERG = np.mean(nc_file.precipitation.values[1:-3].reshape(-1,4,384,352),axis=1)*24
    
    if mask is not None:
        # Create the correct size mask for the IMERG data
        daily_precip_IMERG_mask = np.repeat(
            np.reshape(mask,(1,len(latitude_IMERG),len(longitude_IMERG))),
            days_this_month, axis=0)

        # Create a masked array
        daily_precip_IMERG = ma.masked_array(precip_IMERG, mask=daily_precip_IMERG_mask)
        
    return xr.DataArray(daily_precip_IMERG, dims = ['time','latitude', 'longitude'],
                        coords = {\
                            'time':time_IMERG,
                            'latitude':latitude_IMERG,
                            'longitude':longitude_IMERG,
                        }
                       )


# Load daily mean IFS data for one month (returns mm/day)
# This works with the 1 day lead time data, for temporary use until all IFS data for 2024 is downloaded
def load_daily_mean_IFS_1d_by_month(year,             # Year to load
                                    month,            # Month to load (1-12)
                                    data_dir,         # Where the data is stored
                                    latitude=None,    # List of latitudes to load (None means load all)
                                    longitude=None,   # List of longitudes to load (None means load all)
                                    mask=None         # A mask: True where data is missing, False otherwise
                                   ):
    
    # For the 1 day data this is the case
    num_lead_days = 1

    d_start = datetime(year,month,1)
    d = d_start

    # Load IFS data for the single day forecasts in 2024
    # Temporary until we have the data downloaded from ECMWF.
    file_name = f"{data_dir}/IFS_{d.year}{d.month:02d}{d.day:02d}_00Z.nc"

    # Load the latitude and longitude
    nc_file = nc.Dataset(file_name)
    latitude_full = np.array(nc_file["latitude"][:])
    longitude_full = np.array(nc_file["longitude"][:])
    nc_file.close()

    # Find the latitude indices specifing the region to load
    if (latitude is None):
        latitude = latitude_full
        min_lat_idx = None
        max_lat_idx = None
    else:
        min_lat_idx = np.argmin(np.abs(latitude_full - np.min(latitude)))
        max_lat_idx = np.argmin(np.abs(latitude_full - np.max(latitude))) + 1

    # Find the longitude indices specifing the region to load
    if (longitude is None):
        longitude = longitude_full
        min_lon_idx = None
        max_lon_idx = None
    else:
        min_lon_idx = np.argmin(np.abs(longitude_full - np.min(longitude)))
        max_lon_idx = np.argmin(np.abs(longitude_full - np.max(longitude))) + 1

    # Daily precipitation for the month
    days_this_month = num_days_in_month(year,month)
    daily_precip_IFS = np.zeros((days_this_month,num_lead_days,len(latitude),len(longitude))) * np.nan
        
    # Load each day
    # for day in range(1,days_this_month):  # XXX Start on day 1 because the data is missing for march
    for day in range(days_this_month):

        d = d_start + timedelta(days=day-1)

        # Load the monthly mean ICPAC-WRF data
        file_name = f"{data_dir}/IFS_{d.year}{d.month:02d}{d.day:02d}_00Z.nc"
        nc_file = nc.Dataset(file_name)
        time_IFS = np.array(nc_file["time"][:])
        valid_time_IFS = np.array(nc_file["valid_time"][:4])
        precip_IFS = np.array(nc_file["tp_ensemble_mean"][:4,min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx])
        nc_file.close()

        if (d != datetime(1900,1,1) + timedelta(hours=int(time_IFS))):
            print(f"ERROR: Forecast init times do not align in {d}")

        for i in range(4):
            if (d + timedelta(hours=30+i*6) != datetime(1900,1,1) + timedelta(hours=int(valid_time_IFS[i]))):
                print(f"ERROR: Forecast valid times do not align in {d}")

        # Convert to mm/day and save for later
        daily_precip_IFS[day,0,:,:] = np.sum(precip_IFS, axis=0) * 1000

    if mask is not None:
        # Create the correct size mask for the IFS forecast
        daily_precip_IFS_mask = np.repeat(
            np.repeat(
                np.reshape(mask,(1,1,len(latitude),len(longitude))),
                days_this_month, axis=0),
            num_lead_days, axis=1)

        # Create a masked array
        daily_precip_IFS = ma.masked_array(daily_precip_IFS, mask=daily_precip_IFS_mask)
    
    # XXX Should probably return an xarray dataset instead
    return daily_precip_IFS


# Load daily mean IFS data for one month (returns mm/day)
# Note that the valid times do not change as a function of lead timem, to make
# comparison to measurements easier.
# XXX Throw an error if the final lead time is read
# XXX What happens in Janurary?
def load_daily_mean_IFS_by_month(year,                # Year to load
                                 month,               # Month to load (1-12)
                                 data_dir,            # Where the data is stored
                                 num_lead_days=1,     # Number of forecast lead days to load
                                 lead_time_offset=0,  # Hours since 00:00 UTC before the first lead time
                                 latitude=None,       # List of latitudes to load (None means load all)
                                 longitude=None,      # List of longitudes to load (None means load all)
                                 mask=None            # A mask: True where data is missing, False otherwise
                                ):
    
    # Check that we have a compatible lead_time_offset
    if (lead_time_offset//6 != lead_time_offset/6.0):
        print(f"ERROR: lead_time_offset = {lead_time_offset} but it must be a multiple of 6.")
        return
    
    # Start at 00:00 UTC + lead_time_offset
    d_start = datetime(year,month,1) + timedelta(hours=lead_time_offset)
    d = d_start
    
    # Load full latitude and longitude from the first file
    nc_file = nc.Dataset(f"{data_dir}/{d.year}/tp.nc")
    latitude_full = np.array(nc_file["latitude"][:])
    longitude_full = np.array(nc_file["longitude"][:])
    time_IFS = np.array(nc_file["time"][:])
    valid_time_IFS = np.array(nc_file["fcst_valid_time"][:])
    nc_file.close()
    
    # Find the latitude indices specifing the region to load
    if (latitude is None):
        latitude = latitude_full
        min_lat_idx = None
        max_lat_idx = None
    else:
        min_lat_idx = np.argmin(np.abs(latitude_full - np.min(latitude)))
        max_lat_idx = np.argmin(np.abs(latitude_full - np.max(latitude))) + 1
    
    # Find the longitude indices specifing the region to load
    if (longitude is None):
        longitude = longitude_full
        min_lon_idx = None
        max_lon_idx = None
    else:
        min_lon_idx = np.argmin(np.abs(longitude_full - np.min(longitude)))
        max_lon_idx = np.argmin(np.abs(longitude_full - np.max(longitude))) + 1
    
    # Daily precipitation for the month
    days_this_month = num_days_in_month(year,month)
    daily_precip_IFS = np.zeros((days_this_month,num_lead_days,len(latitude),len(longitude)))
    valid_time_test_all = np.zeros((days_this_month,num_lead_days))
    
    # Find the start date in this file (Usefull for files that hold less than 1 year of data)
    nc_file = nc.Dataset(f"{data_dir}/{d.year}/tp.nc")
    first_time_IFS = np.array(nc_file["time"][0])
    nc_file.close()
    first_time = datetime(1900,1,1) + timedelta(hours=int(first_time_IFS))
    
    # Lead time in days (optional +6 hours).
    for lead_days in range(num_lead_days):
        # We want the valid times to line up with the IMERG data

        # Find the corresponding time indices for data with the same valid time
        time_start_idx = ((datetime(d.year,d.month,d.day) - first_time) - timedelta(days=lead_days)).days
        time_end_idx = ((datetime(d.year,d.month+1,d.day) - first_time) - timedelta(days=lead_days)).days
        valid_time_start_idx = (lead_time_offset//6) + lead_days*4  # Accumulated variables
        valid_time_end_idx = valid_time_start_idx + 4  # +1 day
        
        # Load IFS precipitation for August with this number of lead_days
        nc_file = nc.Dataset(f"{data_dir}/{d.year}/tp.nc")
        precip_IFS = np.array(nc_file["tp_mean"][time_start_idx:time_end_idx,
                                                  valid_time_start_idx:valid_time_end_idx,
                                                  min_lat_idx:max_lat_idx,
                                                  min_lon_idx:max_lon_idx])
        # For testing
        valid_time_test = np.array(nc_file["fcst_valid_time"][time_start_idx:time_end_idx,
                                                              valid_time_start_idx:valid_time_end_idx])
        nc_file.close()

        # Remove negative values
        #precip_IFS[precip_IFS < 0.0] = 0.0

        # Convert from m/6h to mm/day
        precip_IFS = np.mean(precip_IFS, axis=1) * 24*1000/6

        # Save for later
        daily_precip_IFS[:,lead_days,:,:] = precip_IFS
        valid_time_test_all[:,lead_days] = valid_time_test[:,0]
        
    # Check that we have the correct dates
    for lead_days in range(num_lead_days):
        for i in range(days_this_month):
            d_valid = datetime(1900,1,1) + timedelta(hours=int(valid_time_test_all[i,lead_days]))
            if (d_valid != (d + timedelta(days=i))):
                print(f"ERROR: Dates don't line up: lead days = {lead_days} on {d_valid}")
    
    if mask is not None:
        # Create the correct size mask for the IFS forecast
        daily_precip_IFS_mask = np.repeat(
            np.repeat(
                np.reshape(mask,(1,1,len(latitude),len(longitude))),
                days_this_month, axis=0),
            num_lead_days, axis=1)

        # Create a masked array
        daily_precip_IFS = ma.masked_array(daily_precip_IFS, mask=daily_precip_IFS_mask)
    
    # XXX Should probably return an xarray dataset instead
    return daily_precip_IFS


# Load daily mean cGAN data for one month
# XXX Only works for GAN data with 1 day separation
def load_daily_mean_cGAN_by_month(year,                # Year to load
                                  month,               # Month to load (1-12)
                                  data_dir,            # Where the data is stored
                                  num_lead_days=1,     # Number of forecast lead days to load
                                  lead_time_offset=0,  # Hours since 00:00 UTC before the first lead time
                                  full_day_data=True,  # 24h averages (True) or 6h averages (False) XXX Make automatic
                                  latitude=None,       # List of latitudes to load (None means load all)
                                  longitude=None,      # List of longitudes to load (None means load all)
                                  mask=None):          # A mask: True where data is missing, False otherwise

    # The start of the month
    d_start = datetime(year,month,2)

    # Load full latitude and longitude from the first file
    d = d_start
    file_name = f"{data_dir}/GAN_{d.year}{d.month:02d}{d.day-1:02d}_00Z.nc"
    nc_file = nc.Dataset(file_name)
    latitude_cGAN = np.array(nc_file["latitude"][:])
    longitude_cGAN = np.array(nc_file["longitude"][:])
    nc_file.close()

    # Find the latitude indices specifing the region to load
    if (latitude is None):
        latitude = latitude_cGAN
        min_lat_idx = None
        max_lat_idx = None
    else:
        min_lat_idx = np.argmin(np.abs(latitude_cGAN - np.min(latitude)))
        max_lat_idx = np.argmin(np.abs(latitude_cGAN - np.max(latitude))) + 1

    # Find the longitude indices specifing the region to load
    if (longitude is None):
        longitude = longitude_cGAN
        min_lon_idx = None
        max_lon_idx = None
    else:
        min_lon_idx = np.argmin(np.abs(longitude_cGAN - np.min(longitude)))
        max_lon_idx = np.argmin(np.abs(longitude_cGAN - np.max(longitude))) + 1

    # Daily precipitation for the month
    days_this_month = num_days_in_month(year,month)
    daily_precip_cGAN = np.zeros((days_this_month,num_lead_days,len(latitude),len(longitude))) * np.nan
    valid_time_test_all = np.zeros((days_this_month,num_lead_days))

    # The first day of the month
    d = d_start - timedelta(days=num_lead_days-1)

    # XXX Because files are missing
    # d += timedelta(days=1)

    # For each day that data for this month that exists
    while (d < d_start + timedelta(days=days_this_month)):

        # Make the forecasts days line up more closely
        dl = d - timedelta(days=lead_time_offset//24)
        
        file_name = f"{data_dir}/GAN_{dl.year}{dl.month:02d}{dl.day:02d}_00Z.nc"
        # XXX temporary remove (forecast is running)
        if (dl == datetime(2024,4,24)):
            file_name = f"{data_dir}/GAN_{dl.year}{dl.month:02d}{dl.day:02d}_ens50.nc"
        nc_file = nc.Dataset(file_name)
        time_cGAN = np.array(nc_file["time"][0])
        valid_time_cGAN = np.array(nc_file["fcst_valid_time"][0,:])
        precip_cGAN = np.mean(
            np.array(nc_file["precipitation"][0,:,:,min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx]),
            axis=0) * 24  # Convert mm/h to mm/day
        nc_file.close()
        
        # Check that we have the correct dates
        if (dl != datetime(1900,1,1) + timedelta(hours=int(time_cGAN))):
            print(f"ERROR: Times don't match in {file_name}")
        for lead_days in range(num_lead_days):
            if (dl + timedelta(hours=lead_days*24+lead_time_offset) != datetime(1900,1,1) +
                    timedelta(hours=int(valid_time_cGAN[lead_days+1]))):
                print(f"ERROR: Valid times don't match in {file_name}")

        # Save the precipitation for later
        for lead_days in range(num_lead_days):

            # The day of the month corresponding the to current valid time
            d_valid = d + timedelta(days=lead_days)

            # If this day is in the month we are looking at
            if (d_valid >= d_start) and (d_valid < d_start + timedelta(days=days_this_month)):
                
                # Save for further analysis
                if (full_day_data):  # Data is averaged over 24h periods since we start on the second on the month we subtract 2
                    daily_precip_cGAN[d_valid.day-2,lead_days,:,:] = precip_cGAN[lead_days+1,:,:]# idx 0 is valid at 6h, we want idx 1 (30h)
                    valid_time_test_all[d_valid.day-2,lead_days] = valid_time_cGAN[lead_days+1]

                else:  # Data is averaged over 6h periods
                    daily_precip_cGAN[d_valid.day-1,lead_days,:,:] = np.mean(precip_cGAN[lead_days*4:lead_days*4+4,:,:], axis=0)
                    valid_time_test_all[d_valid.day-1,lead_days] = valid_time_cGAN[lead_days]
                    
        # Move to the next date
        d += timedelta(days=1)

    # Check that we have the correct valid times for each day
    for day in range(2,days_this_month+1):
        for lead_days in range(num_lead_days):
            d1 = datetime(1900,1,1) + timedelta(hours=int(valid_time_test_all[day-2,lead_days]))
            d2 = datetime(year,month,day) + timedelta(hours=int(np.mod(lead_time_offset,24)))
            if (d1 != d2):
                print(d1,d2)
                print(f"ERROR: Dates do not line up on {d2}")

    if mask is not None:
        # Create the correct size mask for the IFS forecast
        daily_precip_cGAN_mask = np.repeat(
            np.repeat(
                np.reshape(mask,(1,1,len(latitude),len(longitude))),
                days_this_month, axis=0),
            num_lead_days, axis=1)

        # Create a masked array
        daily_precip_cGAN = ma.masked_array(daily_precip_cGAN, mask=daily_precip_cGAN_mask)

    # XXX Should probably return an xarray dataset instead
    return daily_precip_cGAN, valid_time_test_all


# Load daily mean KMD-WRF data for one month (returns mm/day)
# Note that the valid times do not change as a function of lead timem, to make
# comparison to measurements easier.
# XXX Throw an error if the final lead time is read
# XXX What happens in Janurary?
def load_daily_mean_KMD_WRF_by_month(
    year,                # Year to load
    month,               # Month to load (1-12)
    data_dir,            # Where the data is stored
    num_lead_days=1,     # Number of forecast lead days to load
    lead_time_offset=0,  # Hours since 00:00 UTC before the first lead time
    latitude=None,       # List of latitudes to load (None means load all)
    longitude=None,      # List of longitudes to load (None means load all)
    mask=None            # A mask: True where data is missing, False otherwise
):

    # Check that we have a compatible lead_time_offset
    if (lead_time_offset//3 != lead_time_offset/3.0):
        print(f"ERROR: lead_time_offset = {lead_time_offset} but it must be a multiple of 3.")
        #return

    # Start at 00:00 UTC + lead_time_offset
    d_start = datetime(year,month,1) + timedelta(hours=lead_time_offset)
    d = d_start

    # Daily precipitation for the month
    days_this_month = num_days_in_month(year,month)
    daily_precip_KMD_WRF = np.zeros((days_this_month,num_lead_days,len(latitude),len(longitude)))

    # Load full latitude and longitude from the first file

    # rainnc is the total precipitation
    # It's an accumulated variable, every 3 hours starting from time zero
    # Units are mm in 3 hours

    d_end = d_start + timedelta(days=1)     # Date the forecast ends in the data file
    # XXX Also make code for the other type of file specified
    # d_end = datetime(2024,1,23)  # Date the forecast ends in the data file
    KMD_data_file_name = f"{data_dir}/{d.strftime('%Y%m%d')}_to_{d_end.strftime('%Y%m%d')}_fcst.nc"
    # XXX For testing
    KMD_data_file_name = f"{data_dir}/20240116_to_20240117_fcst.nc"

    nc_file = nc.Dataset(KMD_data_file_name)
    latitude_KMD = np.array(nc_file["lat"][:])
    longitude_KMD = np.array(nc_file["lon"][:])
    nc_file.close()

    # Create an xarray DataArray for the latitudes
    latitude_KMD_da = xr.DataArray(latitude_KMD,
                                     dims="lat",
                                     attrs={"units": "degrees_north"})

    # Create an xarray DataArray for the longitudes
    longitude_KMD_da = xr.DataArray(longitude_KMD,
                                     dims="lon",
                                     attrs={"units": "degrees_east"})
    
    # While we are still looking at this month's data
    d_end_month = d_start + timedelta(days=days_this_month)
    day_idx = 0
    while (d < d_end_month):

        # Precipitation for each lead day
        precip_KMD = np.zeros((num_lead_days, len(latitude_KMD), len(longitude_KMD)))
        for lead_days in range(num_lead_days):


            # d_load is the start time of the forecast
            # d is the valid time we are trying to load
            d_load = d - timedelta(days=(lead_time_offset//24) + lead_days)
            d_end = d_load + timedelta(days=1)     # Date the forecast ends in the data file
            # XXX Also make code for the other type of file specified
            # d_end = datetime(2024,1,23)  # Date the forecast ends in the data file

            # Indices of the start and end forecast times
            start_idx = lead_days*8+lead_time_offset//3
            end_idx = (lead_days+1)*8+lead_time_offset//3

            # Load the KMD-WRF data for the given day
            KMD_data_file_name = f"{data_dir}/{d_load.strftime('%Y%m%d')}_to_{d_end.strftime('%Y%m%d')}_fcst.nc"

            # If the file exists
            #if os.path.isfile(KMD_data_file_name):
            
            # XXX For testing
            if (d_load == datetime(2024,month,16,6)):  # If we want the one file avaliable
                KMD_data_file_name = f"{data_dir}/20240116_to_20240117_fcst.nc"

                # Load as normal
                nc_file = nc.Dataset(KMD_data_file_name)
                precip_accum_KMD_start = np.array(nc_file["rainnc"][start_idx,:,:,:])
                precip_accum_KMD_end = np.array(nc_file["rainnc"][end_idx,:,:,:])
                nc_file.close()

            else:
                print(f"WARNING: {KMD_data_file_name} does not exist. Using np.nan")
                precip_accum_KMD_start = np.nan
                precip_accum_KMD_end = np.nan

            # Convert from accumulated to 24h precipitation
            precip_KMD[lead_days,:,:] = np.squeeze(precip_accum_KMD_end - 
                                                   precip_accum_KMD_start)

        # Create an xarray DataArray for interpolation
        precip_KMD_da = xr.DataArray(precip_KMD,
                                     dims=("lead_days","lat","lon"),
                                     coords={"lead_days": np.arange(num_lead_days),
                                             "lat": latitude_KMD_da,
                                             "lon": longitude_KMD_da},
                                     attrs={"units": "mm/day"})

        # Place the DataArray into a Dataset
        ds_in = xr.Dataset({"precipitation": precip_KMD_da})

        if (latitude is None) and (longitude is None):

            # No interpolation required
            dr_out = ds_in["precipitation"]

        else:  # Interpolate to the new grid

            # Specify the latitude and longitude to interpolate to if not specified
            if (latitude is None):
                latitude = latitude_KMD
            if (longitude is None):
                longitude = longitude_KMD

            # Create the Dataset specification we are interpolating to
            ds_out = xr.Dataset({"lat": (["lat"], latitude, {"units": "degrees_north"}),
                                 "lon": (["lon"], longitude, {"units": "degrees_east"})})

            # Define the regridder function to perform the interpolation
            regridder = xe.Regridder(ds_in, ds_out, "conservative")

            # Regrid the DataArray
            dr_out = regridder(ds_in["precipitation"], keep_attrs=True)

        daily_precip_KMD_WRF[day_idx,:,:,:] = dr_out

        # Move to the next data file
        d += timedelta(days=1)
        day_idx += 1

    if mask is not None:
        # Create the correct size mask for the IFS forecast
        daily_precip_KMD_WRF_mask = np.repeat(
            np.repeat(
                np.reshape(mask,(1,1,len(latitude),len(longitude))),
                days_this_month, axis=0),
            num_lead_days, axis=1)

        # Create a masked array
        daily_precip_KMD_WRF = ma.masked_array(daily_precip_KMD_WRF, mask=daily_precip_KMD_WRF_mask)
    
    # XXX Should probably return an xarray dataset instead
    return daily_precip_KMD_WRF


# Compute some metrics
# Returns a python dictionary with the following keys:
#    Over the region and month and for each lead time:
#       mean_rgn_truth:          Mean true precipitation
#       mean_rgn_forecast:       Mean forecast precipitation
#       std_rgn_truth:           Standard deviation of the true precipitation
#       std_rgn_forecast:        Standard deviation of the forecast precipitation
#       bias_rgn:                Mean bias
#       MSE_rgn:                 Mean Squared Error (Take square root to get the RMSE)
#       MAE_rgn:                 Mean Absolute Error
#       R2_rgn:                  Coefficient of determination
#       anomaly_cov_rgn          Anomaly covariance wrt. ICPAC monthly climatology
#       cov_norm_rgn             Anomaly covariance normalisation
#       anomaly_corr_rgn         Anomaly correlation wrt. ICPAC monthly climatology
#       POD_rgn:                 Probability Of Detection
#       POFA_rgn:                Probability Of False Alarm
#    If each_cell is true then:
#    Over the month and for each grid cell and lead time:
#       mean_truth:              Mean true precipitation
#       mean_forecast:           Mean forecast precipitation
#       std_truth:               Standard deviation of the true precipitation
#       std_forecast:            Standard deviation of the forecast precipitation
#       bias:                    Mean bias
#       MSE:                     Mean Squared Error (Take square root to get the RMSE)
#       MAE:                     Mean Absolute Error
#       R2:                      Coefficient of determination
#       anomaly_cov              Anomaly covariance wrt. ICPAC monthly climatology
#       cov_norm                 Anomaly covariance normalisation
#       anomaly_corr             Anomaly correlation wrt. ICPAC monthly climatology
#    If each_day is true then:
#    Over the region and for each day and lead time:
#       mean_rgn_daily_truth:    Mean true precipitation
#       mean_rgn_daily_forecast: Mean forecast precipitation
#       std_rgn_daily_truth:     Standard deviation of the true precipitation
#       std_rgn_daily_forecast:  Standard deviation of the forecast precipitation
#       bias_rgn_daily:          Mean bias
#       MSE_rgn_daily:           Mean Squared Error (Take square root to get the RMSE)
#       MAE_rgn_daily:           Mean Absolute Error
#       R2_rgn_daily:            Coefficient of determination
#       anomaly_cov_rgn_daily    Anomaly covariance wrt. ICPAC monthly climatology
#       cov_norm_rgn_daily       Anomaly covariance normalisation
#       anomaly_corr_rgn_daily   Anomaly correlation wrt. ICPAC monthly climatology
def compute_deterministic_metrics(precip_truth,     # The precipitation truth we are comparing to
                                  precip_forecast,  # The precipitation forecast we are assessing
                                  region='Kenya',   # The region we are looking at
                                  #latitude, longitude, mask, # XXX Can I remove these?
                                  year=2024,  # XXX Get from forecast or truth
                                  month=1, # XXX Get from forecast or truth
                                  each_cell=True,  # Compute metrics separately for each grid cell
                                  each_day=True    # Compute metrics separately for each day
                                 ):

    # Which country are we looking at.
    latitude, longitude, mask = load_country_mask(region)
    
    # rgn denotes the statistic computed over the region
    # daily denotes the statistic computed daily
    
    # Make a dictionary to hold the results
    dm = {}

    # Number of days in this month
    days_this_month = precip_truth.shape[0]

    # Number of lead days
    num_lead_days = precip_forecast.shape[1]
    
    # Repeat the array to match the dimensions of the forecasts
    truth = np.repeat(np.reshape(precip_truth,
                                 (days_this_month, 1, len(latitude), len(longitude))),
                      num_lead_days, axis=1)
    
    # Summary statistics (allways computed)
        
    # Monthly mean
    dm["mean_rgn_truth"] = np.nanmean(precip_truth)
    dm["mean_rgn_forecast"] = np.nanmean(precip_forecast, axis=(0,2,3))
    
    # Monthly standard deviation
    dm["std_rgn_truth"] = np.nanstd(precip_truth, ddof=1)
    dm["std_rgn_forecast"] = np.nanstd(precip_forecast, ddof=1, axis=(0,2,3))

    # Daily bias
    dm["bias_rgn"] = dm["mean_rgn_forecast"] - dm["mean_rgn_truth"]

    # Daily mean-squared error
    dm["RMSE_rgn"] = np.sqrt(np.nanmean((precip_forecast - truth)**2, axis=(0,2,3)))

    # Daily mean absolute error
    dm["MAE_rgn"] = np.nanmean(np.abs(precip_forecast - truth), axis=(0,2,3))
    
    # Coefficient of determination for the region
    SStot = np.nansum((truth - np.nanmean(truth, axis=0))**2, axis=(0,2,3))
    SSres = np.nansum((truth - precip_forecast)**2, axis=(0,2,3))
    dm["R2_rgn"] = 1 - SSres/SStot
        
    # Load the ICPAC monthly mean climatology
    precip_clim_ICPAC = load_ICPAC_monthly_climatology(ICPAC_climate_data_file, region)
        
    # Compute the  anomaly correlation for this month
    truth_anom = truth - precip_clim_ICPAC[month-1,:,:]
    forecast_anom = precip_forecast - precip_clim_ICPAC[month-1,:,:]
    
    # The anomaly covariance
    dm["anomaly_cov_rgn"] = np.nanmean(truth_anom * forecast_anom, axis=(0,2,3))

    # The anomaly correlation
    dm["cov_norm_rgn"] = np.sqrt(np.nanmean(truth_anom * truth_anom, axis=(0,2,3)) *
                                 np.nanmean(forecast_anom * forecast_anom, axis=(0,2,3)))
    dm["anomaly_corr_rgn"] = dm["anomaly_cov_rgn"] / dm["cov_norm_rgn"]
    
    # Probability of detection (POD) and false alarm (POFA)
    #    XXX POD threshold does not need to be a function of lead time
    #    XXX But there might be multiple thresholds.
    threshold = 50  # mm/day
    dm["POD_threshold"] = threshold
    dm["POD_rgn"] = np.zeros(num_lead_days)
    dm["POFA_rgn"] = np.zeros(num_lead_days)
    for lead_days in range(num_lead_days):

        # IFS and truth both show events above the threshold
        hits = np.nansum((precip_truth > 50) & (precip_forecast[:,lead_days,:,:] > threshold))

        # Truth shows events above the threshold but IFS does not
        misses = np.nansum((precip_truth > 50) & np.logical_not(precip_forecast[:,lead_days,:,:] > threshold))

        # IFS shows events above the threshold but truth does not
        false_alarms = np.nansum(np.logical_not(precip_truth > 50) & (precip_forecast[:,lead_days,:,:] > threshold))

        if (hits + misses > 0):
            dm["POD_rgn"][lead_days] = hits / (hits + misses)
        else:
            dm["POD_rgn"][lead_days] = 0

        if (hits + false_alarms > 0):
            dm["POFA_rgn"][lead_days] = false_alarms / (hits + false_alarms)
        else:
            dm["POFA_rgn"][lead_days] = 0
    
    # Compute statistics separately for each cell
    if each_cell:
    
        # Monthly mean
        dm["mean_truth"] = np.nanmean(precip_truth, axis=0)
        dm["mean_forecast"] = np.nanmean(precip_forecast, axis=0)

        # Monthly standard deviation
        dm["std_truth"] = np.nanstd(precip_truth, ddof=1, axis=0)
        dm["std_forecast"] = np.nanstd(precip_forecast, ddof=1, axis=0)

        # Daily bias
        dm["bias"] = dm["mean_forecast"] - dm["mean_truth"]

        # Daily mean-squared error
        dm["RMSE"] = np.sqrt(np.nanmean((precip_forecast - truth)**2, axis=0))

        # Daily mean absolute error
        dm["MAE"] = np.nanmean(np.abs(precip_forecast - truth), axis=0)

        # Coefficient of determination
        SSres = np.nansum((truth - precip_forecast)**2, axis=0)
        SStot = np.nansum((truth - np.nanmean(truth, axis=0))**2, axis=0)
        dm["R2"] = 1 - SSres/SStot

        # The anomaly covariance
        dm["anomaly_cov"] = np.nanmean(truth_anom * forecast_anom, axis=0)

        # The anomaly correlation
        dm["cov_norm"] = np.sqrt(np.nanmean(truth_anom * truth_anom, axis=0) *
                                 np.nanmean(forecast_anom * forecast_anom, axis=0))
        dm["anomaly_corr"] = dm["anomaly_cov"] / dm["cov_norm"]

    # Compute statistics separately for each day
    if each_day:
    
        # Monthly mean
        dm["mean_rgn_daily_truth"] = np.nanmean(precip_truth, axis=(1,2))
        dm["mean_rgn_daily_forecast"] = np.nanmean(precip_forecast, axis=(2,3))

        # Monthly standard deviation
        dm["std_rgn_daily_truth"] = np.nanstd(precip_truth, ddof=1, axis=(1,2))
        dm["std_rgn_daily_forecast"] = np.nanstd(precip_forecast, ddof=1, axis=(2,3))

        # Daily bias
        dm["bias_rgn_daily"] = np.nanmean(precip_forecast - truth, axis=(2,3))

        # Daily mean-squared error
        dm["RMSE_rgn_daily"] = np.sqrt(np.nanmean((precip_forecast - truth)**2, axis=(2,3)))

        # Daily mean absolute error
        dm["MAE_rgn_daily"] = np.nanmean(np.abs(precip_forecast - truth), axis=(2,3))

        # Coefficient of determination for the region daily
        SStot = np.nansum((truth - np.nanmean(truth, axis=0))**2, axis=(2,3))
        SSres = np.nansum((truth - precip_forecast)**2, axis=(2,3))
        dm["R2_rgn_daily"] = 1 - SSres/SStot
        
        # The anomaly covariance
        dm["anomaly_cov_rgn_daily"] = np.nanmean(truth_anom * forecast_anom, axis=(2,3))

        # The anomaly correlation
        dm["cov_norm_rgn_daily"] = np.sqrt(np.nanmean(truth_anom * truth_anom, axis=(2,3)) *
                                           np.nanmean(forecast_anom * forecast_anom, axis=(2,3)))
        dm["anomaly_corr_rgn_daily"] = dm["anomaly_cov_rgn_daily"] / dm["cov_norm_rgn_daily"]
    
    # Return the dictionary of metrics
    return dm


# Save the deterministic metrics to NetCDF
def save_deterministic_metrics(dm,        # The deterministic metrics to save
                               file_name  # Where to save them
                              ):

    # Number of days, latitudes and longitudes being saved
    days_this_month = len(dm['mean_rgn_daily_truth'])
    num_latitudes = dm['mean_truth'].shape[0]
    num_longitudes = dm['mean_truth'].shape[1]
    num_lead_times = dm['mean_rgn_forecast'].shape[0]

    # Create a new NetCDF file
    rootgrp = nc.Dataset(file_name, "w", format="NETCDF4")

    # Describe where this data comes from
    rootgrp.description = "A selection of deterministic metrics."

    # Create dimensions
    longitude_dim = rootgrp.createDimension("longitude", num_longitudes)
    latitude_dim = rootgrp.createDimension("latitude", num_latitudes)
    time_dim = rootgrp.createDimension("time", days_this_month)
    lead_time_dim = rootgrp.createDimension("lead_time", num_lead_times)
    constant_dim = rootgrp.createDimension("const", 1)
    
    # Monthly and regional mean metrics
    
    # Truth
    const_data = rootgrp.createVariable("mean_rgn_truth", "f8", ("const"), zlib=False)
    const_data[:] = dm['mean_rgn_truth']
    const_data = rootgrp.createVariable("std_rgn_truth", "f8", ("const"), zlib=False)
    const_data[:] = dm['std_rgn_truth']
    
    # Forecast
    keys = ['mean_rgn_forecast','std_rgn_forecast','bias_rgn','RMSE_rgn','MAE_rgn','R2_rgn',
            'anomaly_cov_rgn','cov_norm_rgn','anomaly_corr_rgn','POD_threshold','POD_rgn',
            'POFA_rgn']
    for key in keys:
        const_data = rootgrp.createVariable(key, "f8", ("lead_time"), zlib=False)
        const_data[:] = dm[key]

    # Monthly mean metrics
    
    # Truth
    const_data = rootgrp.createVariable("mean_truth", "f8", ("latitude","longitude"), zlib=False)
    const_data[:] = dm['mean_truth']
    const_data = rootgrp.createVariable("std_truth", "f8", ("latitude","longitude"), zlib=False)
    const_data[:] = dm['std_truth']
    
    # Forecast
    keys = ['mean_forecast','std_forecast','bias','RMSE','MAE','R2',
            'anomaly_cov','cov_norm','anomaly_corr']
    for key in keys:
        const_data = rootgrp.createVariable(key, "f8", ("lead_time","latitude","longitude"), zlib=False)
        const_data[:] = dm[key]

    # Regional mean metrics
    
    # Truth
    const_data = rootgrp.createVariable("mean_rgn_daily_truth", "f8", ("time"), zlib=False)
    const_data[:] = dm['mean_rgn_daily_truth']
    const_data = rootgrp.createVariable("std_rgn_daily_truth", "f8", ("time"), zlib=False)
    const_data[:] = dm['std_rgn_daily_truth']
    
    # Forecast
    keys = ['mean_rgn_daily_forecast','std_rgn_daily_forecast','bias_rgn_daily','RMSE_rgn_daily',
            'MAE_rgn_daily','R2_rgn_daily','anomaly_cov_rgn_daily','cov_norm_rgn_daily',
            'anomaly_corr_rgn_daily']
    for key in keys:
        const_data = rootgrp.createVariable(key, "f8", ("time","lead_time"), zlib=False)
        const_data[:] = dm[key]

    # Close the netCDF file
    rootgrp.close()
    return


# Load deterministic metrics to check
# XXX Check that the year and month saved is correct on loading
def load_deterministic_metrics(file_name):
    
    # Make a dictionary to hold the results
    dm = {}
        
    # Load the monthly mean ICPAC-WRF data
    nc_file = nc.Dataset(file_name)
    
    # The variables saved in the file
    keys = list(nc_file.variables.keys())
    
    # Load each variable
    for key in keys:
        dm[key] = nc_file[key][:]
        
    # Close the netCDF file
    nc_file.close()
    
    return dm

