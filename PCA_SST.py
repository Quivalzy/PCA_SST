import pandas as pd
import xarray as xr
import numpy as np
import sklearn.decomposition as skldc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cftr
import seaborn as sns
import time as timepkg 
sns.set_theme(context='talk', font='serif', palette='colorblind')

HadISST = xr.open_dataset("Modul_5/HadISST_sst.nc")

HadISST_mean = HadISST.sst.where(HadISST.sst > 0).sel(latitude=slice(8,-11), longitude=slice(90,150)).mean(dim=['latitude','longitude'])
HadISST_trend = np.polyfit(x = range(0,len(HadISST_mean.values)),
                           y = HadISST_mean.values,
                           deg = 1,
                           )
HadISST_polyn = np.poly1d(HadISST_trend)
print(HadISST_polyn)
plt.figure(figsize=(12,4))

HadISST_mean.plot()
plt.plot(HadISST.sst.time,
         HadISST_polyn(np.arange(len(HadISST.sst.time))),
         lw=2,
         label=r"$y=({:.2e})x + {:.2f}$".format(HadISST_trend[0], 
                                                HadISST_trend[1])
         )
plt.title('HadISST mean')
plt.legend()
plt.savefig('Mean.png')
plt.close()

HadISST_potong = HadISST.where(HadISST.sst > 0).sst.sel(time=slice('1990','2020'),latitude=slice(8,-11), longitude=slice(90,150))
def detrend_dim(dataArray, dim, deg=1, skipna=True):
  koefisien = dataArray.polyfit(dim=dim, deg=deg, skipna=skipna).polyfit_coefficients
  fit       = xr.polyval(HadISST_potong[dim], koefisien)
  return (dataArray - fit)

HadISST_potong_detrend = detrend_dim(HadISST_potong, 'time', deg=1)

mean = HadISST_potong_detrend.mean(dim=['latitude', 'longitude'])
trend= np.polyfit(x = range(0, len(mean.values)),
                  y = mean.values,
                  deg = 1)
HadISST_polyn = np.poly1d(trend)
print(HadISST_polyn)

plt.figure(figsize=(12,4))

mean.plot()
plt.plot(HadISST_potong.time,
         HadISST_polyn(np.arange(len(HadISST_potong.time))),
         lw=2,
         label=r"$y=({:.2e})x + {:.2f}".format(trend[0], trend[1])
         )
plt.title('HadISST_potong mean detrended')
plt.legend()
plt.savefig('Detrend.png')

HadISST_klimatologi = HadISST_potong.groupby('time.month').mean()
HadISST_klimatologi
HadISST_anomali = HadISST_potong.groupby('time.month') - HadISST_klimatologi

time = HadISST_anomali.time
lat  = HadISST_anomali.latitude
lon  = HadISST_anomali.longitude


HadISST_anomali_reshape = np.reshape(HadISST_anomali.values,
                                     newshape=(len(time), (len(lat) * len(lon)))
                                     )

judul_kolom = pd.MultiIndex.from_product([HadISST_anomali.latitude.values, HadISST_anomali.longitude.values],
                                         names=['latitude', 'longitude']
                                        )

HadISST_anomali_reshape = pd.DataFrame(HadISST_anomali_reshape,
                                       columns= judul_kolom,
                                       index= time
                                       )

HadISST_lokasi_nan_np    = np.isnan(HadISST_anomali_reshape.values).any(axis=0)      
HadISST_lokasi_nan_pd    = (HadISST_anomali_reshape.isna().any())                    
HadISST_anomali_reshape_dropna = HadISST_anomali_reshape.loc[:, ~HadISST_lokasi_nan_pd].T

matrix_input = HadISST_anomali_reshape_dropna.copy(deep=True)
kovariansi = np.cov(matrix_input.T)
evals, evect = np.linalg.eigh(kovariansi)       
urutan_eigen = evals.argsort()[::-1]
evals = evals[urutan_eigen]
evect = evect[:, urutan_eigen]
evals_percent = 100* evals / sum(evals)

proyeksi = np.matmul(matrix_input, evect)
judul_PC = ["PC-" + str(s) for s in range(1, len(proyeksi.columns)+1)]
proyeksi.columns = judul_PC

plt.plot(evals_percent[:20], '-o')
plt.title("Scree-plot")
plt.savefig('Scree.png')

plt.plot(np.cumsum(evals_percent[:20]), '-o')
plt.title("Scree-plot cumulative")
plt.savefig('ScreeC.png')

max_pc = 5
pair_plot = sns.pairplot(proyeksi.iloc[:, :max_pc],
                         diag_kind = 'kde',
                         plot_kws={'s': 2}
                         )
pair_plot.fig.suptitle("Pairplot PC-1 to PC-{}".format(max_pc), y=1.01)
plt.savefig('Pair.png')

start_time = timepkg.time()
proyeksi_reshape_np = HadISST_anomali_reshape.copy(deep=True).values
proyeksi_reshape_np[:, ~HadISST_lokasi_nan_np] = proyeksi.T.values
proyeksi_reshape = proyeksi_reshape_np
proyeksi_reshape_3d = np.reshape(proyeksi_reshape,
                                 newshape=(len(proyeksi.columns),
                                           len(lat),
                                           len(lon),
                                           )
                                 )

HadISST_EOF = xr.Dataset(data_vars= {'projection' : (['EOF','lat','lon'], proyeksi_reshape_3d),
                                     'eigenvector': (['time','EOF'], evect),
                                     'eigenvalue' : (['EOF'], evals),
                                     'var_percent': (['EOF'], evals_percent)
                                     },
                         coords   = {'EOF' : range(1, len(proyeksi.columns)+1),
                                     'lat' : lat.values,
                                     'lon' : lon.values,
                                     'time': time.values
                                     }
                         )

for i_PC in range(1,6):
  fig = plt.figure(figsize=(10,8))

    ## Inputs
  mode_spasial = HadISST_EOF.projection.sel(EOF=i_PC)
  fase_temporal= HadISST_EOF.eigenvector.sel(EOF=i_PC)
  porsi_var    = HadISST_EOF.var_percent.sel(EOF=i_PC)

  level = np.linspace(-10, 10, 21, endpoint=True)

  ax0 = plt.subplot2grid((4,1), (0,0), rowspan=3,
                        projection=ccrs.Robinson(-155)
                        )
  PC_map = mode_spasial.plot.contourf(ax=ax0,
                                      cmap='Spectral_r',
                                      levels=level, extend='both',
                                      transform=ccrs.PlateCarree(),
                                      add_colorbar=False
                                      )
  plt.colorbar(PC_map,
              shrink=0.8,
              pad=0.03
              ).set_label("magnitude")
  ax0.set_title("Spatial pattern of PC-%d" %(i_PC))
  ax0.coastlines(lw=2, color='k', zorder=2)
  ax0.add_feature(cftr.LAND, facecolor='white', zorder=0)
  ax0.gridlines()

  ax1 = plt.subplot2grid((4,1), (3,0), rowspan=1)

  fase_temporal.plot()
  ax1.set_title("Temporal pattern of PC-%d" %(i_PC))
  ax1.grid(True, which='both')
  ax1.set_ylabel('magnitude')
  ax1.set_xlabel('Time')
  ax1.yaxis.set_label_position('right')
  ax1.yaxis.tick_right()

  plt.tight_layout(h_pad=3)
  plt.suptitle("Principal Component %d (Variance: %.4f %%)" %(i_PC, porsi_var), weight='bold', size=18)
  plt.subplots_adjust(top=1)

  plt.savefig("Principal Component %d (Variance: %.4f %%).png" %(i_PC, porsi_var))