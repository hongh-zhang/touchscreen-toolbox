# set up DLC folder
import os
import gdown
import zipfile

if __name__ == "__main__":
    
    if os.path.exists("./touchscreen_toolbox/DLC"):
        print("Folder already exists!")
    
    else:
        # download models
        url = 'https://drive.google.com/uc?id=1Fp0AH54Zj4aly4tmwyuCO0jzuJ0H4GXE'
        output = 'DLC.zip'
        gdown.download(url, output, quiet=False)

        # unzip
        with zipfile.ZipFile('./DLC.zip', 'r') as zip_ref:
            zip_ref.extractall('./touchscreen_toolbox')

        # cleanup
        os.remove("./DLC.zip")
        print("Done!")
    
    # pause
    print("Press the <ENTER> key to continue...")
    input()