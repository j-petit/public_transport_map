"""
Credits go to https://github.com/hkwi/python-geohash
"""

def _encode_i2c(lat, lon, bitlength):
    """
    
    :param lat: 
    :param lon: 
    :param bitlength: 
    :return: 
    """
    digits = '0123'
    r = ''
    while bitlength > 0:
        r += digits[((lat & 1) << 1) + (lon & 1)]
        lat = lat >> 1
        lon = lon >> 1
        bitlength -= 1

    return r[::-1]


def _decode_c2i(treecode):
    """
    
    :param treecode: 
    :return: 
    """
    lat = 0
    lon = 0
    for i in treecode:
        b = ord(i) - 48
        lat = (lat << 1) + int(b / 2)
        lon = (lon << 1) + b % 2

    return (lat, lon, len(treecode))


def encode(lat, lon, precision=18):
    """
    
    :param lat: 
    :param lon: 
    :param precision: 
    :return: 
    """

    b = 1 << precision
    return _encode_i2c(int(b * (lat + 90.0) / 180.0), int(b * (lon + 180.0) / 360.0), precision)


def decode(treecode, delta=False):
    """
    
    :param treecode: 
    :param delta: 
    :return: 
    """
    (lat, lon, bitlength) = _decode_c2i(treecode)
    lat = (lat << 1) + 1
    lon = (lon << 1) + 1
    b = 1 << (bitlength + 1)
    if delta:
        return 180.0 * lat / b - 90.0, 360.0 * lon / b - 180.0, 180.0 / b, 360.0 / b

    return 180.0 * lat / b - 90.0, 360.0 * lon / b - 180.0


def neighbors(treecode):
    """
    
    :param treecode: 
    :return: 
    """
    (lat, lon, bitlength) = _decode_c2i(treecode)
    r = []
    tlat = lat
    for tlon in (lon - 1, lon + 1):
        r.append(_encode_i2c(tlat, tlon, bitlength))

    tlat = lat + 1
    if not tlat >> bitlength:
        for tlon in (lon - 1, lon, lon + 1):
            r.append(_encode_i2c(tlat, tlon, bitlength))

    tlat = lat - 1
    if tlat >= 0:
        for tlon in (lon - 1, lon, lon + 1):
            r.append(_encode_i2c(tlat, tlon, bitlength))

    return r


def expand(treecode):
    """
    
    :param treecode: 
    :return: 
    """
    r = neighbors(treecode)
    r.append(treecode)
    return r