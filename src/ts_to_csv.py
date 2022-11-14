import argparse
from datetime import datetime, timedelta
import pandas as pd


def get_date(start_date, end_date):
    """
    :param start_date: input start date
    :param end_date: input end date
    :return: list type between start and end dates
    """
    from_date = datetime.strptime(start_date, '%Y%m%d')
    to_date = datetime.strptime(end_date, '%Y%m%d')
    delta = timedelta(days=1)

    date_list = []
    while from_date <= to_date:
        date_list.append(from_date.strftime('%Y%m%d'))
        from_date += delta

    return date_list


def get_link_list(dir, start, end):
    """
    :param dir: directory of ts dataset
    :param start: input start date
    :param end: input end date
    :return: link list
    """

    # pd.read_csv()
    file_list = get_date(start, end)
    concat_file = pd.DataFrame(columns=['date', 'link', 'speed'])

    # link_set = set()
    for file in file_list:
        read_file = pd.read_csv('{}/{}/000000_0'.format(dir, file), sep=',', names=['date', 'link', 'speed'],
                                header=None, engine="python", encoding="cp949")
        print("read", file, "file finish")
        concat_file = pd.concat([concat_file, read_file])

    link_list = list(concat_file['link'].unique())
    # link_list = list(link_set)
    link_list.sort()
    # print(link_list)
    time_list = list(concat_file['date'].unique())
    time_len = len(time_list)

    col = ['date']
    csv = pd.DataFrame(time_list, columns=col)
    # for date in file_list:
    k = 0
    for link in link_list:
        link_df = concat_file[concat_file['link'] == link].reset_index(drop=True)
        print(k)
        vel = []
        if len(link_df) == time_len:
            vel = link_df['speed'].tolist()
        else:
            i = 0
            j = 0
            date_temp = link_df['date'].tolist()
            for time in time_list:
                if time in date_temp:
                    j += 1
                    continue
                else:
                    if i == j:
                        vel.append(0)
                        # i += 1
                        # j += 1
                    else:
                        vel = vel + link_df.loc[i:j - 1].speed.tolist()
                        vel.append(0)
                        i = j
            vel = vel + link_df.loc[i:j - 1].speed.tolist()
        # print(vel)
        csv[f'{link}'] = vel
        k += 1

    csv.to_csv("./07_09.csv", index=False)

    # df = pd.read_csv(filename, sep='\s+', names=column_name, header=None, engine="python", encoding="cp949")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='D:/project/Traffic-ETRI/dataset/ts_202107-09',
                        help='directory ')
    parser.add_argument('--start', type=str, default='20210701',
                        help='start date ex)yyyymmdd')
    parser.add_argument('--end', type=str, default='20210930',
                        help='end date ex)yyyymmdd')

    args = parser.parse_args()
    get_link_list(args.dir, args.start, args.end)