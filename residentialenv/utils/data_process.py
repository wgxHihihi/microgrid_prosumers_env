import pandas as pd
import os


def process_ele_data(ele_path, meta_path):
    ele_data = pd.read_csv(ele_path)
    # ele_data['local_15min'] = pd.to_datetime(ele_data['local_15min'])
    meta_data = pd.read_csv(meta_path).iloc[1:, ]
    meta_data['dataid'] = meta_data['dataid'].astype(int)

    no_PV_has_AC = meta_data[(meta_data['pv'] != 'yes')]  # & (meta_data['air1'] != 'yes')]
    # 选择合适的家庭
    legal_id = set(no_PV_has_AC['dataid'])
    # print(legal_id)
    data_id = set(ele_data['dataid'])
    # print(data_id)
    selected_id = list(legal_id.intersection(data_id))
    # print(selected_id)
    for i, id in enumerate(selected_id):
        home_data = ele_data[ele_data['dataid'] == id]

        data_ = pd.DataFrame()
        data_['local_15min'] = home_data['local_15min']
        print(data_['local_15min'])
        data_['ac_sum'] = home_data[['air1', 'air2', 'air3', 'airwindowunit1']].sum(1)
        data_['no_ac_grid'] = home_data['grid'] - data_['ac_sum']
        data_[['air1', 'air2', 'air3', 'airwindowunit1', 'grid']] = home_data[
            ['air1', 'air2', 'air3', 'airwindowunit1', 'grid']]

        data_.to_csv('E:\\嗑盐\\paper\\microgrid\\pecanstreet\\pecan_particial\\15minute_data_austin\\noPV\\home'
                     + str(id) + '.csv', index=False)
        # (data_['local_15min'] >= pd.to_datetime('2018/6/1 00:00:00+00:00')) & (
        #         data_['local_15min'] <= pd.to_datetime('2018/10/31 23:45:00+00:00'))]


def ele_data_train(ele_folder_path, output_path):
    files = os.listdir(ele_folder_path)
    for f in files:
        if not os.path.isdir(f):
            data = pd.read_csv(ele_folder_path + '\\' + f)
            data['local_15min'] = pd.to_datetime(data['local_15min'])
            data = data[(data['local_15min'] >= pd.to_datetime('2018/6/1 00:00:00-05:00')) & (
                    data['local_15min'] <= pd.to_datetime('2018/10/31 23:45:00-05:00'))]
            data.to_csv(output_path + '\\' + f, index=0)


def process_temp_data(temp_data_path):
    temp_data_chunk = pd.read_csv(temp_data_path, chunksize=10000)
    temp_data = []
    for chunk in temp_data_chunk:
        temp_data.append(chunk[chunk['location'].str.contains('Texas')])
    res = pd.concat(temp_data)
    res.to_csv('E:\\嗑盐\\paper\\microgrid\\pecanstreet\\pecan_particial\\Soils_Weather_Data\\Texas\\Texas_temp.csv')
    res = res[res['location'] == 'Texas6']
    # 去重
    res = res[~res['localhour'].duplicated()]
    austin_temp = pd.DataFrame()
    austin_temp['time'] = res['localhour']  # index=pd.to_datetime(res['localhour'])
    austin_temp['T_out'] = (res['temperature'] - 32) * 5 / 9
    temp_series = pd.Series(austin_temp['T_out'].values, index=pd.to_datetime(austin_temp['time']))
    temp_series = temp_series.resample('15min').asfreq()
    temp_series = temp_series.interpolate('linear')
    # austin_15min_temp = austin_temp.resample('15min').asfreq().interpolate('linear')
    temp_series.to_csv(
        'E:\\嗑盐\\paper\\microgrid\\pecanstreet\\pecan_particial\\Soils_Weather_Data\\Texas\\austin_15min_temp.csv')

    # ouput train data
    temp_series[(temp_series.index >= pd.to_datetime('2018/6/1 00:00:00')) & (
            temp_series.index <= pd.to_datetime('2018/10/31 23:45:00'))].to_csv(
        'E:\\嗑盐\\paper\\microgrid\\pecanstreet\\pecan_particial\\Soils_Weather_Data\\Texas\\austin_temp_train.csv')


def process_austin_data(path):
    austin_temp_data = pd.read_csv(path)
    austin_temp_data = austin_temp_data[austin_temp_data['REPORT_TYPE'] == 'CRN05']
    austin_temp = pd.DataFrame()
    austin_temp['time'] = austin_temp_data['DATE']
    austin_temp['T_out'] = (austin_temp_data['HourlyDryBulbTemperature'] - 32) * 5 / 9
    austin_temp.to_csv('E:\\嗑盐\\paper\\microgrid\\pecanstreet\\pecan_particial\\temp.csv')


ele_path = 'E:\\嗑盐\\paper\\microgrid\\pecanstreet\\pecan_particial\\15minute_data_austin\\15minute_data_austin.csv'
meta_path = 'E:\\嗑盐\\paper\\microgrid\\pecanstreet\\pecan_particial\\15minute_data_austin\\metadata.csv'
ele_folder_path = 'E:\\嗑盐\\paper\\microgrid\\pecanstreet\\pecan_particial\\15minute_data_austin\\noPV\\years_data'
output_path = 'E:\\嗑盐\\paper\\microgrid\\pecanstreet\\pecan_particial\\15minute_data_austin\\noPV\\training_data'
# process_ele_data(ele_path, meta_path)

Temp_path = 'E:\\嗑盐\\paper\\microgrid\\pecanstreet\\pecan_particial\\Soils_Weather_Data\\2018-soils-weather.csv'

path = 'E:\\嗑盐\\paper\\microgrid\\pecanstreet\\pecan_particial\\temp_austin.csv'
# ele_data_train(ele_folder_path, output_path)
process_temp_data(Temp_path)
# process_austin_data(path)
