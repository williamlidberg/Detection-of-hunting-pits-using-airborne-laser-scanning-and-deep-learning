#!/bin/bash


# echo "UNET 05m"
# echo "hillshade"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/hillshade/ /workspace/data/logfiles/UNet/05m/hillshade1/trained.h5 /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/hillshade/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/hillshade/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/hillshade/ --output_type=polygon --min_area=30 --min_ratio=-1
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/hillshade/ /workspace/data/logfiles/UNet/05m/hillshade1/objects.csv

# cho "depthinsink"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/depthinsink/ /workspace/data/logfiles/UNet/05m/depthinsink1/trained.h5 /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/depthinsink/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/depthinsink/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/depthinsink/ --output_type=polygon --min_area=30 --min_ratio=-1
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/depthinsink/ /workspace/data/logfiles/UNet/05m/depthinsink1/objects.csv

# echo "elevation_above_pit"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/elevation_above_pit/ /workspace/data/logfiles/UNet/05m/elevation_above_pit1/trained.h5 /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/elevation_above_pit/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/elevation_above_pit/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/elevation_above_pit/ --output_type=polygon --min_area=30 --min_ratio=-1
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/elevation_above_pit/ /workspace/data/logfiles/UNet/05m/elevation_above_pit1/objects.csv

# echo "maxelevationdeviation"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/maxelevationdeviation/ /workspace/data/logfiles/UNet/05m/maxelevationdeviation1/trained.h5 /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/maxelevationdeviation/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/maxelevationdeviation/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/maxelevationdeviation/ --output_type=polygon --min_area=30 --min_ratio=-1
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/maxelevationdeviation/ /workspace/data/logfiles/UNet/05m/maxelevationdeviation1/objects.csv

# echo "maximal_curvature"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/maximal_curvature/ /workspace/data/logfiles/UNet/05m/maximal_curvature1/trained.h5 /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/maximal_curvature/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/maximal_curvature/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/maximal_curvature/ --output_type=polygon --min_area=30 --min_ratio=-1
# python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/maximal_curvature/ /workspace/data/logfiles/UNet/05m/maximal_curvature1/objects.csv

# echo "minimal curvature"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/minimal_curvature/ /workspace/data/logfiles/UNet/05m/minimal_curvature1/trained.h5 /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/minimal_curvature/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/minimal_curvature/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/minimal_curvature/ --output_type=polygon --min_area=30 --min_ratio=-1
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/minimal_curvature/ /workspace/data/logfiles/UNet/05m/minimal_curvature1/objects.csv

# echo "multiscale_stdon"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/multiscale_stdon/ /workspace/data/logfiles/UNet/05m/multiscale_stdon1/trained.h5 /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/multiscale_stdon/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/multiscale_stdon/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/multiscale_stdon/ --output_type=polygon --min_area=30 --min_ratio=-1
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/multiscale_stdon/ /workspace/data/logfiles/UNet/05m/multiscale_stdon1/objects.csv

# echo "multiscaleelevationpercentile"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/multiscaleelevationpercentile/ /workspace/data/logfiles/UNet/05m/multiscaleelevationpercentile1/trained.h5 /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/multiscaleelevationpercentile/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/multiscaleelevationpercentile/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/multiscaleelevationpercentile/ --output_type=polygon --min_area=30 --min_ratio=-1
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/multiscaleelevationpercentile/ /workspace/data/logfiles/UNet/05m/multiscaleelevationpercentile1/objects.csv

# echo "profile_curvature"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/profile_curvature/ /workspace/data/logfiles/UNet/05m/profile_curvature1/trained.h5 /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/profile_curvature/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/profile_curvature/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/profile_curvature/ --output_type=polygon --min_area=30 --min_ratio=-1
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/profile_curvature/ /workspace/data/logfiles/UNet/05m/profile_curvature1/objects.csv

# echo "stdon"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/stdon/ /workspace/data/logfiles/UNet/05m/stdon1/trained.h5 /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/stdon/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_raster/stdon/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/stdon/ --output_type=polygon --min_area=30 --min_ratio=-1
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_UNet/05m/inference_polygon/stdon/ /workspace/data/logfiles/UNet/05m/stdon1/objects.csv




# echo "UNET 1m"
# echo "hillshade"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ /workspace/data/logfiles/UNet/1m/hillshade1/trained.h5 /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/hillshade/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/hillshade/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/hillshade/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "depthinsink"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/logfiles/UNet/1m/depthinsink1/trained.h5 /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/depthinsink/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/depthinsink/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/depthinsink/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "elevation_above_pit"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/elevation_above_pit/ /workspace/data/logfiles/UNet/1m/elevation_above_pit1/trained.h5 /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/elevation_above_pit/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/elevation_above_pit/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/elevation_above_pit/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "maxelevationdeviation"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ /workspace/data/logfiles/UNet/1m/maxelevationdeviation1/trained.h5 /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/maxelevationdeviation/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/maxelevationdeviation/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/maxelevationdeviation/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "maximal_curvature"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/maximal_curvature/ /workspace/data/logfiles/UNet/1m/maximal_curvature1/trained.h5 /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/maximal_curvature/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/maximal_curvature/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/maximal_curvature/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "minimal curvature"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/minimal_curvature/ /workspace/data/logfiles/UNet/1m/minimal_curvature1/trained.h5 /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/minimal_curvature/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/minimal_curvature/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/minimal_curvature/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "multiscale_stdon"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/multiscale_stdon/ /workspace/data/logfiles/UNet/1m/multiscale_stdon1/trained.h5 /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/multiscale_stdon/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/multiscale_stdon/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/multiscale_stdon/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "multiscaleelevationpercentile"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ /workspace/data/logfiles/UNet/1m/multiscaleelevationpercentile1/trained.h5 /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/multiscaleelevationpercentile/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/multiscaleelevationpercentile/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/multiscaleelevationpercentile/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "profile_curvature"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/profile_curvature/ /workspace/data/logfiles/UNet/1m/profile_curvature1/trained.h5 /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/profile_curvature/ UNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_raster/profile_curvature/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/profile_curvature/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "stdon"
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/hillshade/ /workspace/data/logfiles/UNet/1m/hillshade1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/depthinsink/ /workspace/data/logfiles/UNet/1m/depthinsink1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/elevation_above_pit/ /workspace/data/logfiles/UNet/1m/elevation_above_pit1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/maxelevationdeviation/ /workspace/data/logfiles/UNet/1m/maxelevationdeviation1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/maximal_curvature/ /workspace/data/logfiles/UNet/1m/maximal_curvature1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/minimal_curvature/ /workspace/data/logfiles/UNet/1m/minimal_curvature1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/multiscale_stdon/ /workspace/data/logfiles/UNet/1m/multiscale_stdon1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/multiscaleelevationpercentile/ /workspace/data/logfiles/UNet/1m/multiscaleelevationpercentile1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/profile_curvature/ /workspace/data/logfiles/UNet/1m/profile_curvature1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_UNet/1m/inference_polygon/stdon/ /workspace/data/logfiles/UNet/1m/stdon1/objects.csv




# echo "Exception XceptionUNet 05m"
# echo "hillshade"

# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/hillshade/ /workspace/data/logfiles/ExceptionUNet/05m/hillshade1/trained.h5 /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/hillshade/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/hillshade/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/hillshade/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "depthinsink"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/depthinsink/ /workspace/data/logfiles/ExceptionUNet/05m/depthinsink1/trained.h5 /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/depthinsink/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/depthinsink/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/depthinsink/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "elevation_above_pit"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/elevation_above_pit/ /workspace/data/logfiles/ExceptionUNet/05m/elevation_above_pit1/trained.h5 /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/elevation_above_pit/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/elevation_above_pit/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/elevation_above_pit/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "maxelevationdeviation"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/maxelevationdeviation/ /workspace/data/logfiles/ExceptionUNet/05m/maxelevationdeviation1/trained.h5 /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/maxelevationdeviation/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/maxelevationdeviation/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/maxelevationdeviation/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "maximal_curvature"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/maximal_curvature/ /workspace/data/logfiles/ExceptionUNet/05m/maximal_curvature1/trained.h5 /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/maximal_curvature/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/maximal_curvature/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/maximal_curvature/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "minimal curvature"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/minimal_curvature/ /workspace/data/logfiles/ExceptionUNet/05m/minimal_curvature1/trained.h5 /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/minimal_curvature/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/minimal_curvature/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/minimal_curvature/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "multiscale_stdon"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/multiscale_stdon/ /workspace/data/logfiles/ExceptionUNet/05m/multiscale_stdon1/trained.h5 /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/multiscale_stdon/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/multiscale_stdon/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/multiscale_stdon/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "multiscaleelevationpercentile"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/multiscaleelevationpercentile/ /workspace/data/logfiles/ExceptionUNet/05m/multiscaleelevationpercentile1/trained.h5 /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/multiscaleelevationpercentile/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/multiscaleelevationpercentile/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/multiscaleelevationpercentile/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "profile_curvature"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/profile_curvature/ /workspace/data/logfiles/ExceptionUNet/05m/profile_curvature1/trained.h5 /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/profile_curvature/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/profile_curvature/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/profile_curvature/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "stdon"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_05m/testing/stdon/ /workspace/data/logfiles/ExceptionUNet/05m/stdon1/trained.h5 /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/stdon/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_raster/stdon/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/stdon/ --output_type=polygon --min_area=30 --min_ratio=-1


python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/hillshade/ /workspace/data/logfiles/ExceptionUNet/05m/hillshade1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/depthinsink/ /workspace/data/logfiles/ExceptionUNet/05m/depthinsink1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/elevation_above_pit/ /workspace/data/logfiles/ExceptionUNet/05m/elevation_above_pit1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/maxelevationdeviation/ /workspace/data/logfiles/ExceptionUNet/05m/maxelevationdeviation1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/maximal_curvature/ /workspace/data/logfiles/ExceptionUNet/05m/maximal_curvature1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/minimal_curvature/ /workspace/data/logfiles/ExceptionUNet/05m/minimal_curvature1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/multiscale_stdon/ /workspace/data/logfiles/ExceptionUNet/05m/multiscale_stdon1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/multiscaleelevationpercentile/ /workspace/data/logfiles/ExceptionUNet/05m/multiscaleelevationpercentile1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/profile_curvature/ /workspace/data/logfiles/ExceptionUNet/05m/profile_curvature1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_05m/testing/polygon_labels/ /workspace/data/final_data_05m/testing/inference_XceptionUNet/05m/inference_polygon/stdon/ /workspace/data/logfiles/ExceptionUNet/05m/stdon1/objects.csv



# echo "Exception UNet 1m"
# echo "hillshade"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ /workspace/data/logfiles/ExceptionUNet/1m/hillshade1/trained.h5 /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/hillshade/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/hillshade/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/hillshade/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "depthinsink"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/logfiles/ExceptionUNet/1m/depthinsink1/trained.h5 /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/depthinsink/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/depthinsink/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/depthinsink/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "elevation_above_pit"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/elevation_above_pit/ /workspace/data/logfiles/ExceptionUNet/1m/elevation_above_pit1/trained.h5 /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/elevation_above_pit/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/elevation_above_pit/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/elevation_above_pit/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "maxelevationdeviation"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ /workspace/data/logfiles/ExceptionUNet/1m/maxelevationdeviation1/trained.h5 /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/maxelevationdeviation/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/maxelevationdeviation/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/maxelevationdeviation/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "maximal_curvature"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/maximal_curvature/ /workspace/data/logfiles/ExceptionUNet/1m/maximal_curvature1/trained.h5 /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/maximal_curvature/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/maximal_curvature/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/maximal_curvature/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "minimal curvature"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/minimal_curvature/ /workspace/data/logfiles/ExceptionUNet/1m/minimal_curvature1/trained.h5 /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/minimal_curvature/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/minimal_curvature/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/minimal_curvature/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "multiscale_stdon"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/multiscale_stdon/ /workspace/data/logfiles/ExceptionUNet/1m/multiscale_stdon1/trained.h5 /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/multiscale_stdon/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/multiscale_stdon/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/multiscale_stdon/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "multiscaleelevationpercentile"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ /workspace/data/logfiles/ExceptionUNet/1m/multiscaleelevationpercentile1/trained.h5 /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/multiscaleelevationpercentile/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/multiscaleelevationpercentile/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/multiscaleelevationpercentile/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "profile_curvature"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/profile_curvature/ /workspace/data/logfiles/ExceptionUNet/1m/profile_curvature1/trained.h5 /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/profile_curvature/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/profile_curvature/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/profile_curvature/ --output_type=polygon --min_area=30 --min_ratio=-1

# echo "stdon"
# python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/final_data_1m/testing/stdon/ /workspace/data/logfiles/ExceptionUNet/1m/stdon1/trained.h5 /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/stdon/ XceptionUNet --classes 0,1 --tile_size 250
# python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_raster/stdon/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/stdon/ --output_type=polygon --min_area=30 --min_ratio=-1



python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/hillshade/ /workspace/data/logfiles/ExceptionUNet/1m/hillshade1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/depthinsink/ /workspace/data/logfiles/ExceptionUNet/1m/depthinsink1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/elevation_above_pit/ /workspace/data/logfiles/ExceptionUNet/1m/elevation_above_pit1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/maxelevationdeviation/ /workspace/data/logfiles/ExceptionUNet/1m/maxelevationdeviation1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/maximal_curvature/ /workspace/data/logfiles/ExceptionUNet/1m/maximal_curvature1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/minimal_curvature/ /workspace/data/logfiles/ExceptionUNet/1m/minimal_curvature1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/multiscale_stdon/ /workspace/data/logfiles/ExceptionUNet/1m/multiscale_stdon1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/multiscaleelevationpercentile/ /workspace/data/logfiles/ExceptionUNet/1m/multiscaleelevationpercentile1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/profile_curvature/ /workspace/data/logfiles/ExceptionUNet/1m/profile_curvature1/objects.csv
python /workspace/code/semantic_segmentation/evaluate_as_objects.py /workspace/data/final_data_1m/testing/polygon_labels/ /workspace/data/final_data_1m/testing/inference_XceptionUNet/1m/inference_polygon/stdon/ /workspace/data/logfiles/ExceptionUNet/1m/stdon1/objects.csv

