from deeplabcut import convertcsv2h5

if __name__ == '__main__':
    convertcsv2h5('config.yaml', userfeedback=False)
    print('done')
