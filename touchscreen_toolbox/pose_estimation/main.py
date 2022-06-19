# def merge_timestamp(data: pd.DataFrame, timestamp: pd.DataFrame):
#     timestamp = timestamp.convert_dtypes()

#     # align starting time
#     timestamp['time'] += data.time.iloc[0]

#     timestamp['frame'] = utils.sec2frame(timestamp['time'])
#     increment_dup(timestamp, 'frame')

#     merged = data.merge(timestamp.drop('time', axis=1), how='left', on='frame')
#     merged['state_'] = merged['state'].fillna(method='ffill')#.fillna(method='bfill')

#     return merged


# def merge_timestamps(rst_folder, timestamp_file):

#     with h5py.File(timestamp_file, 'r') as timestamps:

#         for csv_file in utils.find_files(rst_folder, 'csv'):

#             print(f"Merging {csv_file}...")

#             csv_path = os.path.join(rst_folder, csv_file)

#             success, vid_info = utils.decode_name(csv_file)

#             if success:
#                 timestamp = pd.DataFrame(timestamps[vid_info['mouse_id']+'/'+vid_info['date']],
#                                          columns=['state','time']).convert_dtypes()
#                 data = pd.read_csv(csv_path)
#                 data = merge_timestamp(data, timestamp)
#                 merged.to_csv(csv_path, index=False)
