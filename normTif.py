# first, normalize the Tif file
from osgeo import gdal, osr
import matplotlib.pyplot as plt


def Geotiff_norm(in_path, out_path):
    input_im = gdal.Open(in_path)
    data = []
    for i in range(input_im.RasterCount):
        input_im_band = input_im.GetRasterBand(i + 1)
        stats = input_im_band.GetStatistics(False, True)
        min_value, max_value = stats[0], stats[1]
        input_im_band_ar = input_im.GetRasterBand(i + 1).ReadAsArray()
        output_im_band_ar = (input_im_band_ar - min_value) / (max_value - min_value)
        data.append(output_im_band_ar.copy())

    output_file = out_path
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(output_file,
                           input_im.RasterXSize,
                           input_im.RasterYSize,
                           input_im.RasterCount,
                           gdal.GDT_Float32)
    for i in range(input_im.RasterCount):
        dst_ds.GetRasterBand(i + 1).WriteArray(data[i])

    dst_ds.SetGeoTransform(input_im.GetGeoTransform())
    wkt = input_im.GetProjection()
    # setting spatial reference of output raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection(srs.ExportToWkt())

    input_im = None
    dst_ds = None


def Geotiff_show(in_path):
    input_im = gdal.Open(in_path)

    for i in range(input_im.RasterCount):
        input_im_band_ar = input_im.GetRasterBand(i + 1).ReadAsArray()
        plt.imshow(input_im_band_ar)
        plt.colorbar()
        plt.show()

    input_im = None
    dst_ds = None


Geotiff_norm('./Dataset_res250/CWSL_resampled.tif', './Dataset_res250/CWSL_norm.tif')
Geotiff_show('./Dataset_res250/CWSL_norm.tif')