import WhiteboxTool as wb

def min_max_scaler(image_array: tuple, min_value: int, max_value: int) -> tuple:

    image_array_scaled = (image_array - min_value) / (max_value - min_value) * 255
    return image_array_scaled


def normalization_nationellhojdmodell_2_0(image_array: tuple) -> tuple:
    #min_value = -322
    #max_value = 2098
    #new_image_array = min_max_scaler(image_array, min_value, max_value)
    return image_array


def normalization_nationellhojdmodell_2_0_hillshade(image_array: tuple) -> tuple:
    new_image_array = wb.hillshade(image_array)
    new_image_array = wb.normalize_hill_shade(new_image_array)
    # min_value = -322
    # max_value = 2098
    # new_image_array = min_max_scaler(image_array, min_value, max_value)
    #whitebox
    return new_image_array


def normalization_nationellhojdmodell_2_0_high_pass_median_filter(image_array: tuple) -> tuple:
    new_image_array = wb.high_pass_median_filter(image_array)
    new_image_array = wb.normalize_high_pass_median_filter(new_image_array)
    return new_image_array


def no_normalization(image_array: tuple) -> tuple:
    return image_array