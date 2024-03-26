import gdown

url = "https://drive.google.com/uc?export=download&confirm=pbef&id=1SqLqv1RCJdiUFnnfIce78M9nJF_KF11U"
output = "OpenEathMap_Mini.zip"
gdown.download(url, output, quiet=False)