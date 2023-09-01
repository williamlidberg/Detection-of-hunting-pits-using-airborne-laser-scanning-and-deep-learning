import exifread

def get_tiff_data(tiff_file):
    tags = exifread.process_file(tiff_file)
    return tags


def get_meter_per_pixel(tags):
    #pixel scale
    return tags['Image Tag 0x830E']

def get_gps_coordinates(tags):
    #Model Tie Point, ie. gps coordinates
    return tags['Image Tag 0x8482']
    