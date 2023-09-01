import requests
import PixelNormalization as PN


def params_ortofoto_2_0(coordinates_list: list, image_size=512, image_year=2021, rendering_rule='KraftigareFargmattnad') -> dict:
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size,image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        compression='LZ77',
        compressionQuality='',
        bandIds='',
        sliceId='',
        mosaicRule='{"where":"ImageYear='+str(image_year)+'"}',
        renderingRule='{"rasterfunction":"'+rendering_rule+'"}', #SKS_VisaRGB KraftigareFargmattnad SKS_VisaCIR
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params


def post_ortofoto_2_0(params: dict) -> dict:
    url = 'https://imgutv.svo.local/arcgis/rest/services/Ortofoto_2_0/ImageServer/exportImage'
    resp = requests.get(url=url, params=params, verify='./cacert.pem')
    data = resp.json()
    return data['href']


def params_ortofoto_1_1(coordinates_list: list, image_size=512, image_year=2022, rendering_rule='KraftigareFargmattnad') -> dict:
    params = dict(
        bbox='{},+{},+{},+{}'.format(coordinates_list[0], coordinates_list[1], coordinates_list[2], coordinates_list[3]),
        bboxSR='3006',
        size='{},{}'.format(image_size,image_size),
        imageSR='',
        time='',
        format='tiff',
        pixelType='UNKNOWN',
        noData='',
        noDataInterpretation='esriNoDataMatchAny',
        interpolation='+RSP_BilinearInterpolation',
        compression='LZ77',
        compressionQuality='',
        bandIds='',
        sliceId='',
        mosaicRule='{"where":"ImageYear='+str(image_year)+'"}',
        renderingRule='{"rasterfunction":"'+rendering_rule+'"}', #SKS_VisaRGB KraftigareFargmattnad SKS_VisaCIR
        adjustAspectRatio='true',
        validateExtent='false',
        lercVersion='1',
        compressionTolerance='',
        f='pjson',
    )
    return params


def post_ortofoto_1_1(params: dict) -> dict:
    url = 'https://imgutv.svo.local/arcgis/rest/services/Ortofoto_1_1/ImageServer/exportImage'
    resp = requests.get(url=url, params=params, verify='./cacert.pem')
    data = resp.json()
    return data['href']


def post_nationellhojdmodell_2_0(params: dict) -> dict:
    url = "https://imgutv.svo.local/arcgis/rest/services/NationellHojdmodell_2_0/ImageServer/exportImage"
    resp = requests.get(url=url, params=params, verify='./cacert.pem')
    data = resp.json()
    return data["href"]


def params_nationellhojdmodell_2_0_raw(coordinates_list: list, image_size=512) -> dict:
    params = dict(
        bbox="{},+{},+{},+{}".format(
            coordinates_list[0],
            coordinates_list[1],
            coordinates_list[2],
            coordinates_list[3],
        ),
        bboxSR="3006",
        size="{},{}".format(image_size, image_size),
        imageSR="",
        time="",
        format="tiff",
        pixelType="UNKNOWN",
        noData="",
        noDataInterpretation="esriNoDataMatchAny",
        interpolation="+RSP_BilinearInterpolation",
        renderingRule='',  # SKS_VisaRGB KraftigareFargmattnad SKS_VisaCIR
        compression="LZ77",
        compressionQuality="",
        bandIds="",
        sliceId="",
        adjustAspectRatio="true",
        validateExtent="false",
        lercVersion="1",
        compressionTolerance="",
        f="pjson",
    )
    return params


def params_nationellhojdmodell_2_0(coordinates_list: list, image_size=512) -> dict:
    params = dict(
        bbox="{},+{},+{},+{}".format(
            coordinates_list[0],
            coordinates_list[1],
            coordinates_list[2],
            coordinates_list[3],
        ),
        bboxSR="3006",
        size="{},{}".format(image_size, image_size),
        imageSR="",
        time="",
        format="tiff",
        pixelType="UNKNOWN",
        noData="",
        noDataInterpretation="esriNoDataMatchAny",
        interpolation="+RSP_BilinearInterpolation",
        renderingRule='{"rasterfunction":"MDTerrangskuggning"}',  # SKS_VisaRGB KraftigareFargmattnad SKS_VisaCIR
        compression="LZ77",
        compressionQuality="",
        bandIds="",
        sliceId="",
        adjustAspectRatio="true",
        validateExtent="false",
        lercVersion="1",
        compressionTolerance="",
        f="pjson",
    )
    return params


# Storing all functions in an object
apis = {
    "nationellhojdmodell_2_0": {
        "params": params_nationellhojdmodell_2_0,
        "post": post_nationellhojdmodell_2_0,
        "normalization": PN.normalization_nationellhojdmodell_2_0,
    },
    "nationellhojdmodell_2_0_hillshade": {
        "params": params_nationellhojdmodell_2_0_raw,
        "post": post_nationellhojdmodell_2_0,
        "normalization": PN.normalization_nationellhojdmodell_2_0_hillshade,
    },
    "nationellhojdmodell_2_0_high_pass_median_filter": {
        "params": params_nationellhojdmodell_2_0_raw,
        "post": post_nationellhojdmodell_2_0,
        "normalization": PN.normalization_nationellhojdmodell_2_0_high_pass_median_filter,
    },
    "ortofoto_2_0": {
        "params": params_ortofoto_2_0,
        "post": post_ortofoto_2_0,
        "normalization": PN.no_normalization,
    },
    "ortofoto_1_1": {
        "params": params_ortofoto_1_1,
        "post": post_ortofoto_1_1,
        "normalization": PN.no_normalization,
    }
}
