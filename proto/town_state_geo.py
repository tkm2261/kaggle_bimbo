# encoding: utf-8

import time
import pandas
from pygeocoder import Geocoder


if __name__ == '__main__':

    data = pandas.read_csv("town_state.csv")

    addresses = data.apply(lambda row: '%s, %s' % (row['Town'], row['State']), axis=1)

    list_lat = []
    list_lon = []
    for address in addresses:
        try:
            results = Geocoder.geocode(address)
            lat, lon = results[0].coordinates
        except Exception:
            lat = 0.
            lon = 0.
        list_lat.append(lat)
        list_lon.append(lon)
        time.sleep(0.5)

    df = pandas.DataFrame(data['Agencia_ID'])
    df['lat'] = list_lat
    df['lon'] = list_lon
    print df
    df.to_csv('agencia_master.csv', index=False)
